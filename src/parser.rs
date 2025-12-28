use crate::types::*;
use std::collections::HashMap;

// ============================================================================
// Parsing Error Type
// ============================================================================

#[derive(Debug)]
pub enum ParseError {
    WastError(String),
    UnsupportedInstruction(String),
    UnsupportedFeature(String),
    InvalidModule(String),
}

impl From<wast::Error> for ParseError {
    fn from(e: wast::Error) -> Self {
        ParseError::WastError(e.to_string())
    }
}

// ============================================================================
// Exports
// ============================================================================

#[derive(Clone, Debug)]
pub enum ExportItem {
    Func(u32),
    Memory(u32),
}

// ============================================================================
// Parse a .wast file and return modules + test assertions
// ============================================================================

pub struct WastTest {
    pub modules: Vec<(Module, HashMap<String, ExportItem>)>,
    pub assertions: Vec<TestAssertion>,
}

#[derive(Debug, Clone)]
pub enum TestAssertion {
    AssertReturn {
        module_idx: usize,
        func_name: String,
        args: Vec<Value>,
        expected: Vec<Value>,
    },
    AssertTrap {
        module_idx: usize,
        func_name: String,
        args: Vec<Value>,
        message: String,
    },
    AssertInvalid {
        message: String,
    },
    AssertMalformed {
        message: String,
    },
}

pub fn parse_wast_file(path: &std::path::Path) -> Result<WastTest, ParseError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| ParseError::WastError(format!("Failed to read file: {}", e)))?;

    parse_wast(&contents)
}

pub fn parse_wast(source: &str) -> Result<WastTest, ParseError> {
    let buf = wast::parser::ParseBuffer::new(source)?;
    let wast: wast::Wast = wast::parser::parse(&buf)?;

    let mut modules: Vec<(Module, HashMap<String, ExportItem>)> = Vec::new();
    let mut assertions: Vec<TestAssertion> = Vec::new();
    let mut current_module_idx: usize = 0;

    for directive in wast.directives {
        match directive {
            wast::WastDirective::Module(mut module) => {
                match parse_quote_module(&mut module) {
                    Ok(parsed) => {
                        modules.push(parsed);
                        current_module_idx = modules.len() - 1;
                    }
                    Err(e) => {
                        // Skip modules we can't parse
                        eprintln!("Warning: couldn't parse module: {:?}", e);
                    }
                }
            }
            wast::WastDirective::AssertReturn { exec, results, .. } => {
                if let Some(assertion) = parse_assert_return(current_module_idx, exec, results)? {
                    assertions.push(assertion);
                }
            }
            wast::WastDirective::AssertTrap { exec, message, .. } => {
                if let Some(assertion) = parse_assert_trap(current_module_idx, exec, message)? {
                    assertions.push(assertion);
                }
            }
            wast::WastDirective::AssertInvalid { message, .. } => {
                assertions.push(TestAssertion::AssertInvalid {
                    message: message.to_string(),
                });
            }
            wast::WastDirective::AssertMalformed { message, .. } => {
                assertions.push(TestAssertion::AssertMalformed {
                    message: message.to_string(),
                });
            }
            _ => {}
        }
    }

    Ok(WastTest { modules, assertions })
}

fn parse_quote_module(
    module: &mut wast::QuoteWat,
) -> Result<(Module, HashMap<String, ExportItem>), ParseError> {
    match module {
        wast::QuoteWat::Wat(wast::Wat::Module(m)) => parse_module(m),
        _ => Err(ParseError::UnsupportedFeature("QuoteModule/Component".to_string())),
    }
}

fn parse_module(
    module: &mut wast::core::Module,
) -> Result<(Module, HashMap<String, ExportItem>), ParseError> {
    let mut types: Vec<RecType> = Vec::new();
    let mut funcs: Vec<Function> = Vec::new();
    let mut mems: Vec<MemoryType> = Vec::new();
    let mut exports: HashMap<String, ExportItem> = HashMap::new();

    let fields = match &module.kind {
        wast::core::ModuleKind::Text(fields) => fields,
        wast::core::ModuleKind::Binary(_) => {
            return Err(ParseError::UnsupportedFeature("Binary module".to_string()));
        }
    };

    // First pass: collect types
    for field in fields.iter() {
        if let wast::core::ModuleField::Type(t) = field {
            let rec_type = parse_type_def(t)?;
            types.push(rec_type);
        }
    }

    // Second pass: parse functions and collect exports
    let mut func_idx = 0u32;
    for field in fields.iter() {
        match field {
            wast::core::ModuleField::Func(f) => {
                let type_idx = resolve_type_use(&f.ty, &mut types)?;
                let func = parse_function(f, type_idx, &types)?;
                funcs.push(func);

                // Handle inline exports
                for name in &f.exports.names {
                    exports.insert(name.to_string(), ExportItem::Func(func_idx));
                }

                func_idx += 1;
            }
            wast::core::ModuleField::Export(e) => {
                let item_idx = resolve_index(&e.item);
                match e.kind {
                    wast::core::ExportKind::Func => {
                        exports.insert(e.name.to_string(), ExportItem::Func(item_idx));
                    }
                    wast::core::ExportKind::Memory => {
                        exports.insert(e.name.to_string(), ExportItem::Memory(item_idx));
                    }
                    _ => {}
                }
            }
            wast::core::ModuleField::Memory(m) => {
                if let Ok(mem_type) = parse_memory_type(m) {
                    mems.push(mem_type);
                }
            }
            _ => {}
        }
    }

    Ok((
        Module {
            types,
            funcs,
            mems,
            start: None,
        },
        exports,
    ))
}

fn resolve_index(idx: &wast::token::Index) -> u32 {
    match idx {
        wast::token::Index::Num(n, _) => *n,
        wast::token::Index::Id(id) => id.name().parse().unwrap_or(0),
    }
}

fn resolve_type_use(
    ty: &wast::core::TypeUse<'_, wast::core::FunctionType<'_>>,
    types: &mut Vec<RecType>,
) -> Result<u32, ParseError> {
    if let Some(idx) = &ty.index {
        return Ok(resolve_index(idx));
    }

    // Create inline type
    let func_type = ty.inline.as_ref().ok_or_else(|| {
        ParseError::InvalidModule("Function has no type".to_string())
    })?;

    let params: Vec<ValueType> = func_type
        .params
        .iter()
        .map(|(_, _, vt)| convert_val_type(vt))
        .collect::<Result<Vec<_>, _>>()?;

    let results: Vec<ValueType> = func_type
        .results
        .iter()
        .map(convert_val_type)
        .collect::<Result<Vec<_>, _>>()?;

    let rec_type = RecType {
        subtypes: vec![SubType {
            composite_type: CompositeType::Func(
                ResultType { types: params },
                ResultType { types: results },
            ),
        }],
    };

    let idx = types.len() as u32;
    types.push(rec_type);
    Ok(idx)
}

fn parse_type_def(t: &wast::core::Type) -> Result<RecType, ParseError> {
    let def = &t.def;

    // Get the function type from the TypeDef's kind field
    let func_type = match &def.kind {
        wast::core::InnerTypeKind::Func(f) => f,
        _ => return Err(ParseError::UnsupportedFeature("Non-func type kind".to_string())),
    };

    let params: Vec<ValueType> = func_type
        .params
        .iter()
        .map(|(_, _, vt)| convert_val_type(vt))
        .collect::<Result<Vec<_>, _>>()?;

    let results: Vec<ValueType> = func_type
        .results
        .iter()
        .map(convert_val_type)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(RecType {
        subtypes: vec![SubType {
            composite_type: CompositeType::Func(
                ResultType { types: params },
                ResultType { types: results },
            ),
        }],
    })
}

fn parse_memory_type(m: &wast::core::Memory) -> Result<MemoryType, ParseError> {
    match &m.kind {
        wast::core::MemoryKind::Normal(ty) => Ok(MemoryType {
            addr_type: AddrType::I32, // TODO: check for memory64
            min: ty.limits.min,
            max: ty.limits.max,
        }),
        _ => Err(ParseError::UnsupportedFeature("Inline memory data".to_string())),
    }
}

fn convert_val_type(vt: &wast::core::ValType) -> Result<ValueType, ParseError> {
    match vt {
        wast::core::ValType::I32 => Ok(ValueType::Number(NumberType::I32)),
        wast::core::ValType::I64 => Ok(ValueType::Number(NumberType::I64)),
        wast::core::ValType::F32 => Ok(ValueType::Number(NumberType::F32)),
        wast::core::ValType::F64 => Ok(ValueType::Number(NumberType::F64)),
        wast::core::ValType::V128 => Ok(ValueType::V128),
        wast::core::ValType::Ref(r) => {
            let heap_type = match &r.heap {
                wast::core::HeapType::Abstract { ty, .. } => match ty {
                    wast::core::AbstractHeapType::Func => HeapType::Func,
                    wast::core::AbstractHeapType::Extern => HeapType::Extern,
                    wast::core::AbstractHeapType::Any => HeapType::Any,
                    _ => HeapType::Any,
                },
                wast::core::HeapType::Concrete(idx) => HeapType::TypeUse(resolve_index(idx)),
            };
            Ok(ValueType::Ref(RefType {
                heap_type,
                nullable: r.nullable,
            }))
        }
    }
}

fn parse_function(
    f: &wast::core::Func,
    type_idx: u32,
    types: &[RecType],
) -> Result<Function, ParseError> {
    let mut locals: Vec<ValueType> = Vec::new();
    let mut local_names: HashMap<String, u32> = HashMap::new();

    match &f.kind {
        wast::core::FuncKind::Inline { locals: func_locals, expression } => {
            // First, get parameter names from the inline type if available
            let mut local_idx = 0u32;
            if let Some(func_type) = &f.ty.inline {
                for (id, _, _vt) in &func_type.params {
                    if let Some(name) = id {
                        local_names.insert(name.name().to_string(), local_idx);
                    }
                    local_idx += 1;
                }
            }

            // Parse locals (declared after parameters)
            for local in func_locals.iter() {
                if let Some(name) = &local.id {
                    local_names.insert(name.name().to_string(), local_idx);
                }
                locals.push(convert_val_type(&local.ty)?);
                local_idx += 1;
            }

            // Convert flat instruction list to our structured format
            let body = convert_instructions(&expression.instrs, type_idx, types, &local_names)?;

            Ok(Function {
                type_idx,
                locals,
                body,
            })
        }
        wast::core::FuncKind::Import(_) => {
            Err(ParseError::UnsupportedFeature("Imported function".to_string()))
        }
    }
}

// ============================================================================
// Instruction conversion - from flat wast format to our structured format
// ============================================================================

struct InstrBuilder {
    body: Vec<Instruction>,
    // Stack of (block_start_idx, is_loop) for handling control flow
    block_stack: Vec<(usize, bool, BlockType)>,
}

impl InstrBuilder {
    fn new() -> Self {
        InstrBuilder {
            body: Vec::new(),
            block_stack: Vec::new(),
        }
    }

    fn emit(&mut self, inst: Instruction) -> NodeIdx {
        let idx = NodeIdx(self.body.len() as u32);
        self.body.push(inst);
        idx
    }

    fn current_idx(&self) -> usize {
        self.body.len()
    }
}

fn convert_instructions(
    instrs: &[wast::core::Instruction],
    func_type_idx: u32,
    types: &[RecType],
    local_names: &HashMap<String, u32>,
) -> Result<Vec<Instruction>, ParseError> {
    let mut builder = InstrBuilder::new();

    // Create an implicit entry block that wraps the whole function
    let entry_block_idx = builder.current_idx();
    builder.block_stack.push((entry_block_idx, false, BlockType::TypeIdx(func_type_idx)));

    // Reserve space for the entry block (we'll fill it in at the end)
    builder.emit(Instruction::Nop); // Placeholder

    // Process instructions
    let mut i = 0;
    while i < instrs.len() {
        let instr = &instrs[i];
        convert_single_instruction(instr, &mut builder, types, local_names)?;
        i += 1;
    }

    // Close any remaining blocks and build the entry block
    finalize_blocks(&mut builder)?;

    Ok(builder.body)
}

/// Resolve an index, using local_names for named references
fn resolve_local_index(idx: &wast::token::Index, local_names: &HashMap<String, u32>) -> u32 {
    match idx {
        wast::token::Index::Num(n, _) => *n,
        wast::token::Index::Id(id) => {
            local_names.get(id.name()).copied().unwrap_or(0)
        }
    }
}

fn convert_single_instruction(
    instr: &wast::core::Instruction,
    builder: &mut InstrBuilder,
    types: &[RecType],
    local_names: &HashMap<String, u32>,
) -> Result<(), ParseError> {
    use wast::core::Instruction as WI;

    match instr {
        // Control flow
        WI::Nop => { builder.emit(Instruction::Nop); }
        WI::Unreachable => { builder.emit(Instruction::Unreachable); }
        WI::Return => { builder.emit(Instruction::Return); }

        WI::Block(bt) => {
            let block_type = convert_block_type(bt, types)?;
            let block_start = builder.current_idx();
            builder.block_stack.push((block_start, false, block_type));
            // Reserve space for the block instruction
            builder.emit(Instruction::Nop);
        }

        WI::Loop(bt) => {
            let block_type = convert_block_type(bt, types)?;
            let block_start = builder.current_idx();
            builder.block_stack.push((block_start, true, block_type));
            builder.emit(Instruction::Nop);
        }

        WI::If(bt) => {
            let block_type = convert_block_type(bt, types)?;
            let block_start = builder.current_idx();
            // We'll handle If specially - push a marker
            builder.block_stack.push((block_start, false, block_type));
            // Reserve space for the If instruction
            builder.emit(Instruction::Nop);
        }

        WI::Else(_) => {
            // Mark the else branch - we'll handle this when we see End
            // For now, just note where else starts
            // This is tricky - we need to track the else position
        }

        WI::End(_) => {
            // Close the current block (this is an inner block with an enclosing block)
            if let Some((block_start, is_loop, block_type)) = builder.block_stack.pop() {
                let block_end = builder.current_idx();

                // Collect all instructions between block_start+1 and block_end
                let mut instrs: Vec<NodeIdx> = Vec::new();
                for idx in (block_start + 1)..block_end {
                    instrs.push(NodeIdx(idx as u32));
                }

                // For inner blocks, next points to after this block in the parent
                // The parent block's instruction list will include this block at some index
                let next_loc = FuncLoc {
                    block_id: NodeIdx(block_start as u32),
                    instr_idx: instrs.len() as u32,
                };

                let block = BlockInst {
                    block_type,
                    instrs,
                    is_loop,
                    next: Some(next_loc),  // Inner blocks have a next location
                };

                // Replace the placeholder at block_start
                builder.body[block_start] = Instruction::Block(Box::new(block));
            }
        }

        WI::Br(idx) => {
            let depth = resolve_index(idx);
            // Find the target block
            let target_idx = if builder.block_stack.len() > depth as usize {
                let stack_idx = builder.block_stack.len() - 1 - depth as usize;
                NodeIdx(builder.block_stack[stack_idx].0 as u32)
            } else {
                NodeIdx(0) // Entry block
            };
            builder.emit(Instruction::Br(target_idx));
        }

        WI::BrIf(idx) => {
            let depth = resolve_index(idx);
            let target_idx = if builder.block_stack.len() > depth as usize {
                let stack_idx = builder.block_stack.len() - 1 - depth as usize;
                NodeIdx(builder.block_stack[stack_idx].0 as u32)
            } else {
                NodeIdx(0)
            };
            builder.emit(Instruction::BrIf(target_idx));
        }

        WI::Call(idx) => { builder.emit(Instruction::Call(resolve_index(idx))); }

        // Locals
        WI::LocalGet(idx) => { builder.emit(Instruction::LocalGet(resolve_local_index(idx, local_names))); }
        WI::LocalSet(idx) => { builder.emit(Instruction::LocalSet(resolve_local_index(idx, local_names))); }
        WI::LocalTee(idx) => { builder.emit(Instruction::LocalTee(resolve_local_index(idx, local_names))); }

        // Stack
        WI::Drop => { builder.emit(Instruction::Drop); }
        WI::Select(_) => { builder.emit(Instruction::Select(None)); }

        // i32 operations
        WI::I32Const(v) => { builder.emit(Instruction::Const(Value::I32(*v))); }
        WI::I32Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::I32 }); }
        WI::I32Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::I32 }); }
        WI::I32Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::I32 }); }
        WI::I32DivS => { builder.emit(Instruction::Binary { op: BinaryOp::DivS, typ: NumberType::I32 }); }
        WI::I32DivU => { builder.emit(Instruction::Binary { op: BinaryOp::DivU, typ: NumberType::I32 }); }
        WI::I32RemS => { builder.emit(Instruction::Binary { op: BinaryOp::RemS, typ: NumberType::I32 }); }
        WI::I32RemU => { builder.emit(Instruction::Binary { op: BinaryOp::RemU, typ: NumberType::I32 }); }
        WI::I32And => { builder.emit(Instruction::Binary { op: BinaryOp::And, typ: NumberType::I32 }); }
        WI::I32Or => { builder.emit(Instruction::Binary { op: BinaryOp::Or, typ: NumberType::I32 }); }
        WI::I32Xor => { builder.emit(Instruction::Binary { op: BinaryOp::Xor, typ: NumberType::I32 }); }
        WI::I32Shl => { builder.emit(Instruction::Binary { op: BinaryOp::Shl, typ: NumberType::I32 }); }
        WI::I32ShrS => { builder.emit(Instruction::Binary { op: BinaryOp::ShrS, typ: NumberType::I32 }); }
        WI::I32ShrU => { builder.emit(Instruction::Binary { op: BinaryOp::ShrU, typ: NumberType::I32 }); }
        WI::I32Rotl => { builder.emit(Instruction::Binary { op: BinaryOp::Rotl, typ: NumberType::I32 }); }
        WI::I32Rotr => { builder.emit(Instruction::Binary { op: BinaryOp::Rotr, typ: NumberType::I32 }); }
        WI::I32Clz => { builder.emit(Instruction::Unary { op: UnaryOp::Clz, typ: NumberType::I32 }); }
        WI::I32Ctz => { builder.emit(Instruction::Unary { op: UnaryOp::Ctz, typ: NumberType::I32 }); }
        WI::I32Popcnt => { builder.emit(Instruction::Unary { op: UnaryOp::Popcnt, typ: NumberType::I32 }); }
        WI::I32Eqz => { builder.emit(Instruction::Unary { op: UnaryOp::Eqz, typ: NumberType::I32 }); }
        WI::I32Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::I32 }); }
        WI::I32Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::I32 }); }
        WI::I32LtS => { builder.emit(Instruction::Compare { op: CompareOp::LtS, typ: NumberType::I32 }); }
        WI::I32LtU => { builder.emit(Instruction::Compare { op: CompareOp::LtU, typ: NumberType::I32 }); }
        WI::I32GtS => { builder.emit(Instruction::Compare { op: CompareOp::GtS, typ: NumberType::I32 }); }
        WI::I32GtU => { builder.emit(Instruction::Compare { op: CompareOp::GtU, typ: NumberType::I32 }); }
        WI::I32LeS => { builder.emit(Instruction::Compare { op: CompareOp::LeS, typ: NumberType::I32 }); }
        WI::I32LeU => { builder.emit(Instruction::Compare { op: CompareOp::LeU, typ: NumberType::I32 }); }
        WI::I32GeS => { builder.emit(Instruction::Compare { op: CompareOp::GeS, typ: NumberType::I32 }); }
        WI::I32GeU => { builder.emit(Instruction::Compare { op: CompareOp::GeU, typ: NumberType::I32 }); }
        WI::I32Extend8S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend8S, typ: NumberType::I32 }); }
        WI::I32Extend16S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend16S, typ: NumberType::I32 }); }

        // i64 operations
        WI::I64Const(v) => { builder.emit(Instruction::Const(Value::I64(*v))); }
        WI::I64Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::I64 }); }
        WI::I64Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::I64 }); }
        WI::I64Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::I64 }); }
        WI::I64DivS => { builder.emit(Instruction::Binary { op: BinaryOp::DivS, typ: NumberType::I64 }); }
        WI::I64DivU => { builder.emit(Instruction::Binary { op: BinaryOp::DivU, typ: NumberType::I64 }); }
        WI::I64RemS => { builder.emit(Instruction::Binary { op: BinaryOp::RemS, typ: NumberType::I64 }); }
        WI::I64RemU => { builder.emit(Instruction::Binary { op: BinaryOp::RemU, typ: NumberType::I64 }); }
        WI::I64And => { builder.emit(Instruction::Binary { op: BinaryOp::And, typ: NumberType::I64 }); }
        WI::I64Or => { builder.emit(Instruction::Binary { op: BinaryOp::Or, typ: NumberType::I64 }); }
        WI::I64Xor => { builder.emit(Instruction::Binary { op: BinaryOp::Xor, typ: NumberType::I64 }); }
        WI::I64Shl => { builder.emit(Instruction::Binary { op: BinaryOp::Shl, typ: NumberType::I64 }); }
        WI::I64ShrS => { builder.emit(Instruction::Binary { op: BinaryOp::ShrS, typ: NumberType::I64 }); }
        WI::I64ShrU => { builder.emit(Instruction::Binary { op: BinaryOp::ShrU, typ: NumberType::I64 }); }
        WI::I64Rotl => { builder.emit(Instruction::Binary { op: BinaryOp::Rotl, typ: NumberType::I64 }); }
        WI::I64Rotr => { builder.emit(Instruction::Binary { op: BinaryOp::Rotr, typ: NumberType::I64 }); }
        WI::I64Clz => { builder.emit(Instruction::Unary { op: UnaryOp::Clz, typ: NumberType::I64 }); }
        WI::I64Ctz => { builder.emit(Instruction::Unary { op: UnaryOp::Ctz, typ: NumberType::I64 }); }
        WI::I64Popcnt => { builder.emit(Instruction::Unary { op: UnaryOp::Popcnt, typ: NumberType::I64 }); }
        WI::I64Eqz => { builder.emit(Instruction::Unary { op: UnaryOp::Eqz, typ: NumberType::I64 }); }
        WI::I64Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::I64 }); }
        WI::I64Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::I64 }); }
        WI::I64LtS => { builder.emit(Instruction::Compare { op: CompareOp::LtS, typ: NumberType::I64 }); }
        WI::I64LtU => { builder.emit(Instruction::Compare { op: CompareOp::LtU, typ: NumberType::I64 }); }
        WI::I64GtS => { builder.emit(Instruction::Compare { op: CompareOp::GtS, typ: NumberType::I64 }); }
        WI::I64GtU => { builder.emit(Instruction::Compare { op: CompareOp::GtU, typ: NumberType::I64 }); }
        WI::I64LeS => { builder.emit(Instruction::Compare { op: CompareOp::LeS, typ: NumberType::I64 }); }
        WI::I64LeU => { builder.emit(Instruction::Compare { op: CompareOp::LeU, typ: NumberType::I64 }); }
        WI::I64GeS => { builder.emit(Instruction::Compare { op: CompareOp::GeS, typ: NumberType::I64 }); }
        WI::I64GeU => { builder.emit(Instruction::Compare { op: CompareOp::GeU, typ: NumberType::I64 }); }
        WI::I64Extend8S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend8S, typ: NumberType::I64 }); }
        WI::I64Extend16S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend16S, typ: NumberType::I64 }); }
        WI::I64Extend32S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend32S, typ: NumberType::I64 }); }

        // Conversions
        WI::I32WrapI64 => { builder.emit(Instruction::Convert { op: ConvertOp::Wrap, from: NumberType::I64, to: NumberType::I32 }); }
        WI::I64ExtendI32S => { builder.emit(Instruction::Convert { op: ConvertOp::ExtendS, from: NumberType::I32, to: NumberType::I64 }); }
        WI::I64ExtendI32U => { builder.emit(Instruction::Convert { op: ConvertOp::ExtendU, from: NumberType::I32, to: NumberType::I64 }); }

        // f32 operations
        WI::F32Const(v) => { builder.emit(Instruction::Const(Value::F32(f32::from_bits(v.bits)))); }
        WI::F32Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::F32 }); }
        WI::F32Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::F32 }); }
        WI::F32Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::F32 }); }
        WI::F32Div => { builder.emit(Instruction::Binary { op: BinaryOp::Div, typ: NumberType::F32 }); }
        WI::F32Neg => { builder.emit(Instruction::Unary { op: UnaryOp::Neg, typ: NumberType::F32 }); }
        WI::F32Abs => { builder.emit(Instruction::Unary { op: UnaryOp::Abs, typ: NumberType::F32 }); }
        WI::F32Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::F32 }); }
        WI::F32Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::F32 }); }
        WI::F32Lt => { builder.emit(Instruction::Compare { op: CompareOp::Lt, typ: NumberType::F32 }); }
        WI::F32Gt => { builder.emit(Instruction::Compare { op: CompareOp::Gt, typ: NumberType::F32 }); }
        WI::F32Le => { builder.emit(Instruction::Compare { op: CompareOp::Le, typ: NumberType::F32 }); }
        WI::F32Ge => { builder.emit(Instruction::Compare { op: CompareOp::Ge, typ: NumberType::F32 }); }

        // f64 operations
        WI::F64Const(v) => { builder.emit(Instruction::Const(Value::F64(f64::from_bits(v.bits)))); }
        WI::F64Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::F64 }); }
        WI::F64Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::F64 }); }
        WI::F64Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::F64 }); }
        WI::F64Div => { builder.emit(Instruction::Binary { op: BinaryOp::Div, typ: NumberType::F64 }); }
        WI::F64Neg => { builder.emit(Instruction::Unary { op: UnaryOp::Neg, typ: NumberType::F64 }); }
        WI::F64Abs => { builder.emit(Instruction::Unary { op: UnaryOp::Abs, typ: NumberType::F64 }); }
        WI::F64Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::F64 }); }
        WI::F64Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::F64 }); }
        WI::F64Lt => { builder.emit(Instruction::Compare { op: CompareOp::Lt, typ: NumberType::F64 }); }
        WI::F64Gt => { builder.emit(Instruction::Compare { op: CompareOp::Gt, typ: NumberType::F64 }); }
        WI::F64Le => { builder.emit(Instruction::Compare { op: CompareOp::Le, typ: NumberType::F64 }); }
        WI::F64Ge => { builder.emit(Instruction::Compare { op: CompareOp::Ge, typ: NumberType::F64 }); }

        // Memory operations
        WI::I32Load(m) => { builder.emit(Instruction::Load { mem_arg: convert_mem_arg(m), val_type: NumberType::I32 }); }
        WI::I64Load(m) => { builder.emit(Instruction::Load { mem_arg: convert_mem_arg(m), val_type: NumberType::I64 }); }
        WI::F32Load(m) => { builder.emit(Instruction::Load { mem_arg: convert_mem_arg(m), val_type: NumberType::F32 }); }
        WI::F64Load(m) => { builder.emit(Instruction::Load { mem_arg: convert_mem_arg(m), val_type: NumberType::F64 }); }
        WI::I32Store(m) => { builder.emit(Instruction::Store { mem_arg: convert_mem_arg(m), val_type: NumberType::I32 }); }
        WI::I64Store(m) => { builder.emit(Instruction::Store { mem_arg: convert_mem_arg(m), val_type: NumberType::I64 }); }
        WI::F32Store(m) => { builder.emit(Instruction::Store { mem_arg: convert_mem_arg(m), val_type: NumberType::F32 }); }
        WI::F64Store(m) => { builder.emit(Instruction::Store { mem_arg: convert_mem_arg(m), val_type: NumberType::F64 }); }
        WI::MemorySize(m) => { builder.emit(Instruction::MemorySize(resolve_index(&m.mem))); }

        other => {
            return Err(ParseError::UnsupportedInstruction(format!("{:?}", other)));
        }
    }

    Ok(())
}

fn finalize_blocks(builder: &mut InstrBuilder) -> Result<(), ParseError> {
    // Close remaining blocks (should just be the entry block)
    // The entry block has next: None to indicate implicit return when it completes
    while let Some((block_start, is_loop, block_type)) = builder.block_stack.pop() {
        let block_end = builder.current_idx();

        let mut instrs: Vec<NodeIdx> = Vec::new();
        for idx in (block_start + 1)..block_end {
            instrs.push(NodeIdx(idx as u32));
        }

        let block = BlockInst {
            block_type,
            instrs,
            is_loop,
            next: None,  // Entry block has no next - falls through to implicit return
        };

        builder.body[block_start] = Instruction::Block(Box::new(block));
    }
    Ok(())
}

fn convert_block_type(
    bt: &wast::core::BlockType,
    _types: &[RecType],
) -> Result<BlockType, ParseError> {
    match &bt.ty {
        wast::core::TypeUse { index: Some(idx), .. } => {
            Ok(BlockType::TypeIdx(resolve_index(idx)))
        }
        wast::core::TypeUse { inline: Some(ft), .. } => {
            if ft.params.is_empty() {
                if ft.results.is_empty() {
                    Ok(BlockType::None)
                } else if ft.results.len() == 1 {
                    Ok(BlockType::Value(convert_val_type(&ft.results[0])?))
                } else {
                    Err(ParseError::UnsupportedFeature("Multi-value block".to_string()))
                }
            } else {
                Err(ParseError::UnsupportedFeature("Block with params".to_string()))
            }
        }
        _ => Ok(BlockType::None),
    }
}

fn convert_mem_arg(m: &wast::core::MemArg) -> MemArg {
    MemArg {
        mem: resolve_index(&m.memory),
        align: m.align,
        offset: m.offset,
    }
}

// ============================================================================
// Parse test assertions
// ============================================================================

fn parse_assert_return(
    module_idx: usize,
    exec: wast::WastExecute,
    results: Vec<wast::WastRet>,
) -> Result<Option<TestAssertion>, ParseError> {
    let (func_name, args) = match exec {
        wast::WastExecute::Invoke(invoke) => {
            let args: Vec<Value> = invoke.args.iter().filter_map(convert_wast_arg).collect();
            (invoke.name.to_string(), args)
        }
        _ => return Ok(None),
    };

    let expected: Vec<Value> = results.iter().filter_map(|r| match r {
        wast::WastRet::Core(arg) => convert_core_ret(arg),
        _ => None,
    }).collect();

    Ok(Some(TestAssertion::AssertReturn {
        module_idx,
        func_name,
        args,
        expected,
    }))
}

fn parse_assert_trap(
    module_idx: usize,
    exec: wast::WastExecute,
    message: &str,
) -> Result<Option<TestAssertion>, ParseError> {
    let (func_name, args) = match exec {
        wast::WastExecute::Invoke(invoke) => {
            let args: Vec<Value> = invoke.args.iter().filter_map(convert_wast_arg).collect();
            (invoke.name.to_string(), args)
        }
        _ => return Ok(None),
    };

    Ok(Some(TestAssertion::AssertTrap {
        module_idx,
        func_name,
        args,
        message: message.to_string(),
    }))
}

fn convert_wast_arg(arg: &wast::WastArg) -> Option<Value> {
    match arg {
        wast::WastArg::Core(core) => convert_core_arg(core),
        _ => None,
    }
}

fn convert_core_arg(arg: &wast::core::WastArgCore) -> Option<Value> {
    match arg {
        wast::core::WastArgCore::I32(v) => Some(Value::I32(*v)),
        wast::core::WastArgCore::I64(v) => Some(Value::I64(*v)),
        wast::core::WastArgCore::F32(v) => Some(Value::F32(f32::from_bits(v.bits))),
        wast::core::WastArgCore::F64(v) => Some(Value::F64(f64::from_bits(v.bits))),
        _ => None,
    }
}

fn convert_core_ret(arg: &wast::core::WastRetCore) -> Option<Value> {
    match arg {
        wast::core::WastRetCore::I32(v) => Some(Value::I32(*v)),
        wast::core::WastRetCore::I64(v) => Some(Value::I64(*v)),
        wast::core::WastRetCore::F32(v) => match v {
            wast::core::NanPattern::Value(f) => Some(Value::F32(f32::from_bits(f.bits))),
            _ => None, // Skip NaN patterns for now
        },
        wast::core::WastRetCore::F64(v) => match v {
            wast::core::NanPattern::Value(f) => Some(Value::F64(f64::from_bits(f.bits))),
            _ => None,
        },
        _ => None,
    }
}
