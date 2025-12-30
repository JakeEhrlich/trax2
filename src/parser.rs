use crate::types::*;
use std::collections::HashMap;
use wasmparser::{Parser, Payload, ValType as WpValType, Operator};

// ============================================================================
// Parsing Error Type
// ============================================================================

#[derive(Debug)]
pub enum ParseError {
    WastError(String),
    UnsupportedInstruction(String),
    UnsupportedFeature(String),
    InvalidModule(String),
    BinaryParseError(String),
}

impl From<wast::Error> for ParseError {
    fn from(e: wast::Error) -> Self {
        ParseError::WastError(e.to_string())
    }
}

impl From<wasmparser::BinaryReaderError> for ParseError {
    fn from(e: wasmparser::BinaryReaderError) -> Self {
        ParseError::BinaryParseError(e.to_string())
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

// ============================================================================
// Binary Module Parsing (via wasmparser)
// ============================================================================

fn parse_binary_module(bytes: &[u8]) -> Result<(Module, HashMap<String, ExportItem>), ParseError> {
    let mut types: Vec<RecType> = Vec::new();
    let mut funcs: Vec<Function> = Vec::new();
    let mut exports: HashMap<String, ExportItem> = HashMap::new();

    // Track function type indices (from import + local functions)
    let mut func_type_indices: Vec<u32> = Vec::new();
    let mut num_imported_funcs: u32 = 0;

    // Temporary storage for function bodies
    let mut func_bodies: Vec<wasmparser::FunctionBody> = Vec::new();

    let parser = Parser::new(0);
    for payload in parser.parse_all(bytes) {
        let payload = payload?;
        match payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader {
                    let rec_group = rec_group?;
                    for sub_type in rec_group.into_types() {
                        if let wasmparser::CompositeInnerType::Func(func_type) = sub_type.composite_type.inner {
                            let params: Vec<ValueType> = func_type.params()
                                .iter()
                                .map(|vt| convert_wp_val_type(*vt))
                                .collect::<Result<Vec<_>, _>>()?;
                            let results: Vec<ValueType> = func_type.results()
                                .iter()
                                .map(|vt| convert_wp_val_type(*vt))
                                .collect::<Result<Vec<_>, _>>()?;

                            types.push(RecType {
                                subtypes: vec![SubType {
                                    composite_type: CompositeType::Func(
                                        ResultType { types: params },
                                        ResultType { types: results },
                                    ),
                                }],
                            });
                        }
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import?;
                    if let wasmparser::TypeRef::Func(type_idx) = import.ty {
                        func_type_indices.push(type_idx);
                        num_imported_funcs += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for type_idx in reader {
                    let type_idx = type_idx?;
                    func_type_indices.push(type_idx);
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export?;
                    match export.kind {
                        wasmparser::ExternalKind::Func => {
                            exports.insert(export.name.to_string(), ExportItem::Func(export.index));
                        }
                        wasmparser::ExternalKind::Memory => {
                            exports.insert(export.name.to_string(), ExportItem::Memory(export.index));
                        }
                        _ => {}
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                func_bodies.push(body);
            }
            _ => {}
        }
    }

    // Now parse function bodies
    for (i, body) in func_bodies.into_iter().enumerate() {
        let func_idx = num_imported_funcs + i as u32;
        let type_idx = func_type_indices.get(func_idx as usize)
            .copied()
            .ok_or_else(|| ParseError::InvalidModule("Function without type".to_string()))?;

        // Get locals
        let mut locals: Vec<ValueType> = Vec::new();
        let locals_reader = body.get_locals_reader()?;
        for local in locals_reader {
            let (count, val_type) = local?;
            let vt = convert_wp_val_type(val_type)?;
            for _ in 0..count {
                locals.push(vt.clone());
            }
        }

        // Parse function body
        let ops_reader = body.get_operators_reader()?;
        let func = parse_binary_function(type_idx, locals, ops_reader)?;
        funcs.push(func);
    }

    Ok((
        Module {
            types,
            funcs,
            mems: vec![],
            start: None,
        },
        exports,
    ))
}

fn convert_wp_val_type(vt: WpValType) -> Result<ValueType, ParseError> {
    match vt {
        WpValType::I32 => Ok(ValueType::Number(NumberType::I32)),
        WpValType::I64 => Ok(ValueType::Number(NumberType::I64)),
        WpValType::F32 => Ok(ValueType::Number(NumberType::F32)),
        WpValType::F64 => Ok(ValueType::Number(NumberType::F64)),
        _ => Err(ParseError::UnsupportedFeature(format!("Value type {:?}", vt))),
    }
}

fn parse_binary_function(
    type_idx: u32,
    locals: Vec<ValueType>,
    ops_reader: wasmparser::OperatorsReader,
) -> Result<Function, ParseError> {
    let mut builder = InstrBuilder::new();

    // Create entry block
    builder.emit(Instruction::Nop); // Placeholder for entry block
    builder.block_stack.push((0, BlockKind::Block, BlockType::None));

    for op in ops_reader {
        let op = op?;
        parse_binary_operator(&mut builder, op)?;
    }

    // Finalize remaining blocks
    finalize_blocks(&mut builder)?;

    Ok(Function {
        type_idx,
        locals,
        body: builder.body,
    })
}

fn parse_binary_operator(builder: &mut InstrBuilder, op: Operator) -> Result<(), ParseError> {
    use Operator as Op;

    match op {
        Op::Unreachable => { builder.emit(Instruction::Unreachable); },
        Op::Nop => { builder.emit(Instruction::Nop); },

        Op::Block { blockty } => {
            let block_type = convert_wp_block_type(blockty)?;
            let block_start = builder.current_idx();
            builder.emit(Instruction::Nop); // Placeholder
            builder.block_stack.push((block_start, BlockKind::Block, block_type));
        }
        Op::Loop { blockty } => {
            let block_type = convert_wp_block_type(blockty)?;
            let block_start = builder.current_idx();
            builder.emit(Instruction::Nop); // Placeholder
            builder.block_stack.push((block_start, BlockKind::Loop, block_type));
        }
        Op::If { blockty } => {
            let block_type = convert_wp_block_type(blockty)?;
            let if_start = builder.current_idx();
            builder.emit(Instruction::Nop); // Placeholder for If
            builder.block_stack.push((if_start, BlockKind::If { else_idx: None }, block_type));
        }
        Op::Else => {
            // Mark the else position in the current if block
            let idx = builder.current_idx();
            if let Some((_, BlockKind::If { else_idx }, _)) = builder.block_stack.last_mut() {
                *else_idx = Some(idx);
            }
        }
        Op::End => {
            if let Some((block_start, kind, block_type)) = builder.block_stack.pop() {
                let block_end = builder.current_idx();

                // Calculate next location - after this block in the parent
                let next_loc = if let Some((parent_start, _, _)) = builder.block_stack.last() {
                    let mut parent_instr_idx = 0u32;
                    let mut idx = *parent_start + 1;
                    while idx < block_start {
                        if let Some(&(_, end)) = builder.consumed_ranges.iter().find(|&&(s, _)| s == idx) {
                            idx = end;
                        } else {
                            parent_instr_idx += 1;
                            idx += 1;
                        }
                    }
                    FuncLoc {
                        block_id: NodeIdx(*parent_start as u32),
                        instr_idx: parent_instr_idx + 1,
                    }
                } else {
                    FuncLoc { block_id: NodeIdx(0), instr_idx: 0 }
                };

                builder.mark_consumed(block_start, block_end);

                match kind {
                    BlockKind::If { else_idx } => {
                        let else_start = else_idx.unwrap_or(block_end);
                        let then_instrs = builder.collect_instrs(block_start, else_start);
                        // For else, don't skip the first instruction (no block header to skip)
                        let else_instrs: Vec<NodeIdx> = (else_start..block_end)
                            .filter(|i| !builder.consumed_ranges.iter().any(|(s, e)| *i >= *s && *i < *e))
                            .map(|i| NodeIdx(i as u32))
                            .collect();

                        let then_block_idx = builder.body.len();
                        let then_block = BlockInst {
                            block_type: block_type.clone(),
                            instrs: then_instrs,
                            is_loop: false,
                            next: if builder.block_stack.is_empty() { None } else { Some(next_loc.clone()) },
                        };
                        builder.body.push(Instruction::Block(Box::new(then_block)));

                        let else_block_idx = builder.body.len();
                        let else_block = BlockInst {
                            block_type: block_type.clone(),
                            instrs: else_instrs,
                            is_loop: false,
                            next: if builder.block_stack.is_empty() { None } else { Some(next_loc) },
                        };
                        builder.body.push(Instruction::Block(Box::new(else_block)));

                        // Exclude helper blocks from all parent instruction lists
                        builder.exclude_index(then_block_idx);
                        builder.exclude_index(else_block_idx);

                        let if_inst = IfInst {
                            block_type,
                            then_block: NodeIdx(then_block_idx as u32),
                            else_block: NodeIdx(else_block_idx as u32),
                            next: FuncLoc { block_id: NodeIdx(0), instr_idx: 0 },
                        };
                        builder.body[block_start] = Instruction::If(Box::new(if_inst));
                    }
                    BlockKind::Block | BlockKind::Loop => {
                        let instrs = builder.collect_instrs(block_start, block_end);
                        let is_loop = matches!(kind, BlockKind::Loop);
                        let block = BlockInst {
                            block_type,
                            instrs,
                            is_loop,
                            next: if builder.block_stack.is_empty() { None } else { Some(next_loc) },
                        };
                        builder.body[block_start] = Instruction::Block(Box::new(block));
                    }
                }
            }
        }

        Op::Br { relative_depth } => {
            let target_idx = resolve_binary_branch_target(builder, relative_depth);
            builder.emit(Instruction::Br(target_idx));
        }
        Op::BrIf { relative_depth } => {
            let target_idx = resolve_binary_branch_target(builder, relative_depth);
            builder.emit(Instruction::BrIf(target_idx));
        }
        Op::BrTable { targets } => {
            let labels: Vec<NodeIdx> = targets.targets()
                .map(|t| t.map(|d| resolve_binary_branch_target(builder, d)))
                .collect::<Result<Vec<_>, _>>()?;
            let default = resolve_binary_branch_target(builder, targets.default());
            builder.emit(Instruction::BrTable { labels, default });
        }

        Op::Return => { builder.emit(Instruction::Return); },
        Op::Call { function_index } => { builder.emit(Instruction::Call(function_index)); },

        Op::Drop => { builder.emit(Instruction::Drop); },
        Op::Select => { builder.emit(Instruction::Select(None)); },

        Op::LocalGet { local_index } => { builder.emit(Instruction::LocalGet(local_index)); },
        Op::LocalSet { local_index } => { builder.emit(Instruction::LocalSet(local_index)); },
        Op::LocalTee { local_index } => { builder.emit(Instruction::LocalTee(local_index)); },

        // i32 operations
        Op::I32Const { value } => { builder.emit(Instruction::Const(Value::I32(value))); },
        Op::I32Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::I32 }); },
        Op::I32Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::I32 }); },
        Op::I32Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::I32 }); },
        Op::I32DivS => { builder.emit(Instruction::Binary { op: BinaryOp::DivS, typ: NumberType::I32 }); },
        Op::I32DivU => { builder.emit(Instruction::Binary { op: BinaryOp::DivU, typ: NumberType::I32 }); },
        Op::I32RemS => { builder.emit(Instruction::Binary { op: BinaryOp::RemS, typ: NumberType::I32 }); },
        Op::I32RemU => { builder.emit(Instruction::Binary { op: BinaryOp::RemU, typ: NumberType::I32 }); },
        Op::I32And => { builder.emit(Instruction::Binary { op: BinaryOp::And, typ: NumberType::I32 }); },
        Op::I32Or => { builder.emit(Instruction::Binary { op: BinaryOp::Or, typ: NumberType::I32 }); },
        Op::I32Xor => { builder.emit(Instruction::Binary { op: BinaryOp::Xor, typ: NumberType::I32 }); },
        Op::I32Shl => { builder.emit(Instruction::Binary { op: BinaryOp::Shl, typ: NumberType::I32 }); },
        Op::I32ShrS => { builder.emit(Instruction::Binary { op: BinaryOp::ShrS, typ: NumberType::I32 }); },
        Op::I32ShrU => { builder.emit(Instruction::Binary { op: BinaryOp::ShrU, typ: NumberType::I32 }); },
        Op::I32Rotl => { builder.emit(Instruction::Binary { op: BinaryOp::Rotl, typ: NumberType::I32 }); },
        Op::I32Rotr => { builder.emit(Instruction::Binary { op: BinaryOp::Rotr, typ: NumberType::I32 }); },
        Op::I32Clz => { builder.emit(Instruction::Unary { op: UnaryOp::Clz, typ: NumberType::I32 }); },
        Op::I32Ctz => { builder.emit(Instruction::Unary { op: UnaryOp::Ctz, typ: NumberType::I32 }); },
        Op::I32Popcnt => { builder.emit(Instruction::Unary { op: UnaryOp::Popcnt, typ: NumberType::I32 }); },
        Op::I32Eqz => { builder.emit(Instruction::Unary { op: UnaryOp::Eqz, typ: NumberType::I32 }); },
        Op::I32Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::I32 }); },
        Op::I32Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::I32 }); },
        Op::I32LtS => { builder.emit(Instruction::Compare { op: CompareOp::LtS, typ: NumberType::I32 }); },
        Op::I32LtU => { builder.emit(Instruction::Compare { op: CompareOp::LtU, typ: NumberType::I32 }); },
        Op::I32GtS => { builder.emit(Instruction::Compare { op: CompareOp::GtS, typ: NumberType::I32 }); },
        Op::I32GtU => { builder.emit(Instruction::Compare { op: CompareOp::GtU, typ: NumberType::I32 }); },
        Op::I32LeS => { builder.emit(Instruction::Compare { op: CompareOp::LeS, typ: NumberType::I32 }); },
        Op::I32LeU => { builder.emit(Instruction::Compare { op: CompareOp::LeU, typ: NumberType::I32 }); },
        Op::I32GeS => { builder.emit(Instruction::Compare { op: CompareOp::GeS, typ: NumberType::I32 }); },
        Op::I32GeU => { builder.emit(Instruction::Compare { op: CompareOp::GeU, typ: NumberType::I32 }); },
        Op::I32Extend8S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend8S, typ: NumberType::I32 }); },
        Op::I32Extend16S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend16S, typ: NumberType::I32 }); },

        // i64 operations
        Op::I64Const { value } => { builder.emit(Instruction::Const(Value::I64(value))); },
        Op::I64Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::I64 }); },
        Op::I64Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::I64 }); },
        Op::I64Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::I64 }); },
        Op::I64DivS => { builder.emit(Instruction::Binary { op: BinaryOp::DivS, typ: NumberType::I64 }); },
        Op::I64DivU => { builder.emit(Instruction::Binary { op: BinaryOp::DivU, typ: NumberType::I64 }); },
        Op::I64RemS => { builder.emit(Instruction::Binary { op: BinaryOp::RemS, typ: NumberType::I64 }); },
        Op::I64RemU => { builder.emit(Instruction::Binary { op: BinaryOp::RemU, typ: NumberType::I64 }); },
        Op::I64And => { builder.emit(Instruction::Binary { op: BinaryOp::And, typ: NumberType::I64 }); },
        Op::I64Or => { builder.emit(Instruction::Binary { op: BinaryOp::Or, typ: NumberType::I64 }); },
        Op::I64Xor => { builder.emit(Instruction::Binary { op: BinaryOp::Xor, typ: NumberType::I64 }); },
        Op::I64Shl => { builder.emit(Instruction::Binary { op: BinaryOp::Shl, typ: NumberType::I64 }); },
        Op::I64ShrS => { builder.emit(Instruction::Binary { op: BinaryOp::ShrS, typ: NumberType::I64 }); },
        Op::I64ShrU => { builder.emit(Instruction::Binary { op: BinaryOp::ShrU, typ: NumberType::I64 }); },
        Op::I64Rotl => { builder.emit(Instruction::Binary { op: BinaryOp::Rotl, typ: NumberType::I64 }); },
        Op::I64Rotr => { builder.emit(Instruction::Binary { op: BinaryOp::Rotr, typ: NumberType::I64 }); },
        Op::I64Clz => { builder.emit(Instruction::Unary { op: UnaryOp::Clz, typ: NumberType::I64 }); },
        Op::I64Ctz => { builder.emit(Instruction::Unary { op: UnaryOp::Ctz, typ: NumberType::I64 }); },
        Op::I64Popcnt => { builder.emit(Instruction::Unary { op: UnaryOp::Popcnt, typ: NumberType::I64 }); },
        Op::I64Eqz => { builder.emit(Instruction::Unary { op: UnaryOp::Eqz, typ: NumberType::I64 }); },
        Op::I64Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::I64 }); },
        Op::I64Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::I64 }); },
        Op::I64LtS => { builder.emit(Instruction::Compare { op: CompareOp::LtS, typ: NumberType::I64 }); },
        Op::I64LtU => { builder.emit(Instruction::Compare { op: CompareOp::LtU, typ: NumberType::I64 }); },
        Op::I64GtS => { builder.emit(Instruction::Compare { op: CompareOp::GtS, typ: NumberType::I64 }); },
        Op::I64GtU => { builder.emit(Instruction::Compare { op: CompareOp::GtU, typ: NumberType::I64 }); },
        Op::I64LeS => { builder.emit(Instruction::Compare { op: CompareOp::LeS, typ: NumberType::I64 }); },
        Op::I64LeU => { builder.emit(Instruction::Compare { op: CompareOp::LeU, typ: NumberType::I64 }); },
        Op::I64GeS => { builder.emit(Instruction::Compare { op: CompareOp::GeS, typ: NumberType::I64 }); },
        Op::I64GeU => { builder.emit(Instruction::Compare { op: CompareOp::GeU, typ: NumberType::I64 }); },
        Op::I64Extend8S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend8S, typ: NumberType::I64 }); },
        Op::I64Extend16S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend16S, typ: NumberType::I64 }); },
        Op::I64Extend32S => { builder.emit(Instruction::Unary { op: UnaryOp::Extend32S, typ: NumberType::I64 }); },

        // f32 operations
        Op::F32Const { value } => { builder.emit(Instruction::Const(Value::F32(f32::from_bits(value.bits())))); },
        Op::F32Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::F32 }); },
        Op::F32Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::F32 }); },
        Op::F32Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::F32 }); },
        Op::F32Div => { builder.emit(Instruction::Binary { op: BinaryOp::Div, typ: NumberType::F32 }); },
        Op::F32Min => { builder.emit(Instruction::Binary { op: BinaryOp::Min, typ: NumberType::F32 }); },
        Op::F32Max => { builder.emit(Instruction::Binary { op: BinaryOp::Max, typ: NumberType::F32 }); },
        Op::F32Copysign => { builder.emit(Instruction::Binary { op: BinaryOp::Copysign, typ: NumberType::F32 }); },
        Op::F32Abs => { builder.emit(Instruction::Unary { op: UnaryOp::Abs, typ: NumberType::F32 }); },
        Op::F32Neg => { builder.emit(Instruction::Unary { op: UnaryOp::Neg, typ: NumberType::F32 }); },
        Op::F32Sqrt => { builder.emit(Instruction::Unary { op: UnaryOp::Sqrt, typ: NumberType::F32 }); },
        Op::F32Ceil => { builder.emit(Instruction::Unary { op: UnaryOp::Ceil, typ: NumberType::F32 }); },
        Op::F32Floor => { builder.emit(Instruction::Unary { op: UnaryOp::Floor, typ: NumberType::F32 }); },
        Op::F32Trunc => { builder.emit(Instruction::Unary { op: UnaryOp::Trunc, typ: NumberType::F32 }); },
        Op::F32Nearest => { builder.emit(Instruction::Unary { op: UnaryOp::Nearest, typ: NumberType::F32 }); },
        Op::F32Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::F32 }); },
        Op::F32Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::F32 }); },
        Op::F32Lt => { builder.emit(Instruction::Compare { op: CompareOp::Lt, typ: NumberType::F32 }); },
        Op::F32Gt => { builder.emit(Instruction::Compare { op: CompareOp::Gt, typ: NumberType::F32 }); },
        Op::F32Le => { builder.emit(Instruction::Compare { op: CompareOp::Le, typ: NumberType::F32 }); },
        Op::F32Ge => { builder.emit(Instruction::Compare { op: CompareOp::Ge, typ: NumberType::F32 }); },

        // f64 operations
        Op::F64Const { value } => { builder.emit(Instruction::Const(Value::F64(f64::from_bits(value.bits())))); },
        Op::F64Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::F64 }); },
        Op::F64Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::F64 }); },
        Op::F64Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::F64 }); },
        Op::F64Div => { builder.emit(Instruction::Binary { op: BinaryOp::Div, typ: NumberType::F64 }); },
        Op::F64Min => { builder.emit(Instruction::Binary { op: BinaryOp::Min, typ: NumberType::F64 }); },
        Op::F64Max => { builder.emit(Instruction::Binary { op: BinaryOp::Max, typ: NumberType::F64 }); },
        Op::F64Copysign => { builder.emit(Instruction::Binary { op: BinaryOp::Copysign, typ: NumberType::F64 }); },
        Op::F64Abs => { builder.emit(Instruction::Unary { op: UnaryOp::Abs, typ: NumberType::F64 }); },
        Op::F64Neg => { builder.emit(Instruction::Unary { op: UnaryOp::Neg, typ: NumberType::F64 }); },
        Op::F64Sqrt => { builder.emit(Instruction::Unary { op: UnaryOp::Sqrt, typ: NumberType::F64 }); },
        Op::F64Ceil => { builder.emit(Instruction::Unary { op: UnaryOp::Ceil, typ: NumberType::F64 }); },
        Op::F64Floor => { builder.emit(Instruction::Unary { op: UnaryOp::Floor, typ: NumberType::F64 }); },
        Op::F64Trunc => { builder.emit(Instruction::Unary { op: UnaryOp::Trunc, typ: NumberType::F64 }); },
        Op::F64Nearest => { builder.emit(Instruction::Unary { op: UnaryOp::Nearest, typ: NumberType::F64 }); },
        Op::F64Eq => { builder.emit(Instruction::Compare { op: CompareOp::Eq, typ: NumberType::F64 }); },
        Op::F64Ne => { builder.emit(Instruction::Compare { op: CompareOp::Ne, typ: NumberType::F64 }); },
        Op::F64Lt => { builder.emit(Instruction::Compare { op: CompareOp::Lt, typ: NumberType::F64 }); },
        Op::F64Gt => { builder.emit(Instruction::Compare { op: CompareOp::Gt, typ: NumberType::F64 }); },
        Op::F64Le => { builder.emit(Instruction::Compare { op: CompareOp::Le, typ: NumberType::F64 }); },
        Op::F64Ge => { builder.emit(Instruction::Compare { op: CompareOp::Ge, typ: NumberType::F64 }); },

        // Conversions
        Op::I32WrapI64 => { builder.emit(Instruction::Convert { op: ConvertOp::Wrap, from: NumberType::I64, to: NumberType::I32 }); },
        Op::I64ExtendI32S => { builder.emit(Instruction::Convert { op: ConvertOp::ExtendS, from: NumberType::I32, to: NumberType::I64 }); },
        Op::I64ExtendI32U => { builder.emit(Instruction::Convert { op: ConvertOp::ExtendU, from: NumberType::I32, to: NumberType::I64 }); },
        Op::I32TruncF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F32, to: NumberType::I32 }); },
        Op::I32TruncF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F32, to: NumberType::I32 }); },
        Op::I32TruncF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F64, to: NumberType::I32 }); },
        Op::I32TruncF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F64, to: NumberType::I32 }); },
        Op::I64TruncF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F32, to: NumberType::I64 }); },
        Op::I64TruncF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F32, to: NumberType::I64 }); },
        Op::I64TruncF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F64, to: NumberType::I64 }); },
        Op::I64TruncF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F64, to: NumberType::I64 }); },
        Op::I32TruncSatF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F32, to: NumberType::I32 }); },
        Op::I32TruncSatF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F32, to: NumberType::I32 }); },
        Op::I32TruncSatF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F64, to: NumberType::I32 }); },
        Op::I32TruncSatF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F64, to: NumberType::I32 }); },
        Op::I64TruncSatF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F32, to: NumberType::I64 }); },
        Op::I64TruncSatF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F32, to: NumberType::I64 }); },
        Op::I64TruncSatF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F64, to: NumberType::I64 }); },
        Op::I64TruncSatF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F64, to: NumberType::I64 }); },
        Op::F32ConvertI32S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I32, to: NumberType::F32 }); },
        Op::F32ConvertI32U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I32, to: NumberType::F32 }); },
        Op::F32ConvertI64S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I64, to: NumberType::F32 }); },
        Op::F32ConvertI64U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I64, to: NumberType::F32 }); },
        Op::F64ConvertI32S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I32, to: NumberType::F64 }); },
        Op::F64ConvertI32U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I32, to: NumberType::F64 }); },
        Op::F64ConvertI64S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I64, to: NumberType::F64 }); },
        Op::F64ConvertI64U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I64, to: NumberType::F64 }); },
        Op::F32DemoteF64 => { builder.emit(Instruction::Convert { op: ConvertOp::Demote, from: NumberType::F64, to: NumberType::F32 }); },
        Op::F64PromoteF32 => { builder.emit(Instruction::Convert { op: ConvertOp::Promote, from: NumberType::F32, to: NumberType::F64 }); },
        Op::I32ReinterpretF32 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::F32, to: NumberType::I32 }); },
        Op::I64ReinterpretF64 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::F64, to: NumberType::I64 }); },
        Op::F32ReinterpretI32 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::I32, to: NumberType::F32 }); },
        Op::F64ReinterpretI64 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::I64, to: NumberType::F64 }); },

        other => {
            return Err(ParseError::UnsupportedInstruction(format!("{:?}", other)));
        }
    }

    Ok(())
}

fn resolve_binary_branch_target(builder: &InstrBuilder, relative_depth: u32) -> NodeIdx {
    if builder.block_stack.len() > relative_depth as usize {
        let stack_idx = builder.block_stack.len() - 1 - relative_depth as usize;
        NodeIdx(builder.block_stack[stack_idx].0 as u32)
    } else {
        NodeIdx(0) // Entry block
    }
}

fn convert_wp_block_type(bt: wasmparser::BlockType) -> Result<BlockType, ParseError> {
    match bt {
        wasmparser::BlockType::Empty => Ok(BlockType::None),
        wasmparser::BlockType::Type(vt) => {
            Ok(BlockType::Value(convert_wp_val_type(vt)?))
        }
        wasmparser::BlockType::FuncType(idx) => {
            Ok(BlockType::TypeIdx(idx))
        }
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
        wast::core::ModuleKind::Binary(bytes) => {
            // Concatenate all byte slices and parse as binary
            let mut all_bytes: Vec<u8> = Vec::new();
            for slice in bytes.iter() {
                all_bytes.extend_from_slice(slice);
            }
            return parse_binary_module(&all_bytes);
        }
    };

    // First pass: collect types and build name-to-index map
    let mut type_names: HashMap<String, u32> = HashMap::new();
    for field in fields.iter() {
        if let wast::core::ModuleField::Type(t) = field {
            let type_idx = types.len() as u32;
            if let Some(id) = &t.id {
                type_names.insert(id.name().to_string(), type_idx);
            }
            let rec_type = parse_type_def(t)?;
            types.push(rec_type);
        }
    }

    // Second pass: parse functions and collect exports
    let mut func_idx = 0u32;
    for field in fields.iter() {
        match field {
            wast::core::ModuleField::Func(f) => {
                let type_idx = resolve_type_use(&f.ty, &mut types, &type_names)?;
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

fn resolve_index_with_names(idx: &wast::token::Index, names: &HashMap<String, u32>) -> u32 {
    match idx {
        wast::token::Index::Num(n, _) => *n,
        wast::token::Index::Id(id) => {
            names.get(id.name()).copied().unwrap_or(0)
        }
    }
}

fn resolve_type_use(
    ty: &wast::core::TypeUse<'_, wast::core::FunctionType<'_>>,
    types: &mut Vec<RecType>,
    type_names: &HashMap<String, u32>,
) -> Result<u32, ParseError> {
    if let Some(idx) = &ty.index {
        return Ok(resolve_index_with_names(idx, type_names));
    }

    // Get inline type, or create empty function type if none specified
    let (params, results): (Vec<ValueType>, Vec<ValueType>) = match ty.inline.as_ref() {
        Some(func_type) => {
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

            (params, results)
        }
        None => {
            // No type specified - create void -> void function type
            (vec![], vec![])
        }
    };

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
            // Get the number of parameters from the type definition
            let mut local_idx = 0u32;

            // First try inline type for param names
            if let Some(func_type) = &f.ty.inline {
                for (id, _, _vt) in &func_type.params {
                    if let Some(name) = id {
                        local_names.insert(name.name().to_string(), local_idx);
                    }
                    local_idx += 1;
                }
            } else {
                // No inline type - get param count from the type definition
                if let Some(rec_type) = types.get(type_idx as usize) {
                    if let Some(sub_type) = rec_type.subtypes.first() {
                        if let CompositeType::Func(params, _) = &sub_type.composite_type {
                            local_idx = params.types.len() as u32;
                        }
                    }
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

enum BlockKind {
    Block,
    Loop,
    If { else_idx: Option<usize> },
}

struct InstrBuilder {
    body: Vec<Instruction>,
    // Stack of (block_start_idx, kind, block_type) for handling control flow
    block_stack: Vec<(usize, BlockKind, BlockType)>,
    // Ranges consumed by nested blocks at each nesting level
    // When a block finishes, its range is added here so parent doesn't include its contents
    consumed_ranges: Vec<(usize, usize)>,  // (start, end) - exclusive end
    // Indices that should be completely excluded from all instruction lists (helper blocks)
    excluded_indices: std::collections::HashSet<usize>,
}

impl InstrBuilder {
    fn new() -> Self {
        InstrBuilder {
            body: Vec::new(),
            block_stack: Vec::new(),
            consumed_ranges: Vec::new(),
            excluded_indices: std::collections::HashSet::new(),
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

    /// Collect direct child instruction indices for a block, skipping consumed ranges
    /// but including the consumed block itself (not its contents)
    fn collect_instrs(&self, block_start: usize, block_end: usize) -> Vec<NodeIdx> {
        let mut instrs = Vec::new();
        let mut idx = block_start + 1;  // Skip the block instruction itself
        while idx < block_end {
            // Skip completely excluded indices (helper blocks for If)
            if self.excluded_indices.contains(&idx) {
                idx += 1;
                continue;
            }
            // Check if this index is the start of a consumed range (nested block)
            if let Some(&(start, end)) = self.consumed_ranges.iter().find(|&&(s, _)| s == idx) {
                // Include the nested block itself, but skip its contents
                instrs.push(NodeIdx(start as u32));
                idx = end;
            } else {
                instrs.push(NodeIdx(idx as u32));
                idx += 1;
            }
        }
        instrs
    }

    /// Mark a range as consumed by a nested block
    fn mark_consumed(&mut self, start: usize, end: usize) {
        self.consumed_ranges.push((start, end));
    }

    /// Exclude an index completely from all instruction lists (for helper blocks)
    fn exclude_index(&mut self, idx: usize) {
        self.excluded_indices.insert(idx);
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
    builder.block_stack.push((entry_block_idx, BlockKind::Block, BlockType::TypeIdx(func_type_idx)));

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
            builder.block_stack.push((block_start, BlockKind::Block, block_type));
            // Reserve space for the block instruction
            builder.emit(Instruction::Nop);
        }

        WI::Loop(bt) => {
            let block_type = convert_block_type(bt, types)?;
            let block_start = builder.current_idx();
            builder.block_stack.push((block_start, BlockKind::Loop, block_type));
            builder.emit(Instruction::Nop);
        }

        WI::If(bt) => {
            let block_type = convert_block_type(bt, types)?;
            let block_start = builder.current_idx();
            builder.block_stack.push((block_start, BlockKind::If { else_idx: None }, block_type));
            // Reserve space for the If instruction
            builder.emit(Instruction::Nop);
        }

        WI::Else(_) => {
            // Mark the else position in the current if block
            let idx = builder.current_idx();
            if let Some((_, BlockKind::If { else_idx }, _)) = builder.block_stack.last_mut() {
                *else_idx = Some(idx);
            }
        }

        WI::End(_) => {
            // Close the current block
            if let Some((block_start, kind, block_type)) = builder.block_stack.pop() {
                let block_end = builder.current_idx();

                // Calculate next location for branching
                let next_loc = if let Some((parent_start, _, _)) = builder.block_stack.last() {
                    let mut parent_instr_idx = 0u32;
                    let mut idx = *parent_start + 1;
                    while idx < block_start {
                        if let Some(&(_, end)) = builder.consumed_ranges.iter().find(|&&(s, _)| s == idx) {
                            idx = end;
                        } else {
                            parent_instr_idx += 1;
                            idx += 1;
                        }
                    }
                    FuncLoc {
                        block_id: NodeIdx(*parent_start as u32),
                        instr_idx: parent_instr_idx + 1,
                    }
                } else {
                    FuncLoc { block_id: NodeIdx(0), instr_idx: 0 }
                };

                builder.mark_consumed(block_start, block_end);

                match kind {
                    BlockKind::If { else_idx } => {
                        // Create then and else blocks
                        let else_start = else_idx.unwrap_or(block_end);
                        let then_instrs = builder.collect_instrs(block_start, else_start);
                        // For else, don't skip the first instruction (no block header to skip)
                        let else_instrs: Vec<NodeIdx> = (else_start..block_end)
                            .map(|i| NodeIdx(i as u32))
                            .collect();

                        // Create then block
                        let then_block_idx = builder.body.len();
                        let then_block = BlockInst {
                            block_type: block_type.clone(),
                            instrs: then_instrs,
                            is_loop: false,
                            next: if builder.block_stack.is_empty() { None } else { Some(next_loc.clone()) },
                        };
                        builder.body.push(Instruction::Block(Box::new(then_block)));

                        // Create else block
                        let else_block_idx = builder.body.len();
                        let else_block = BlockInst {
                            block_type: block_type.clone(),
                            instrs: else_instrs,
                            is_loop: false,
                            next: if builder.block_stack.is_empty() { None } else { Some(next_loc) },
                        };
                        builder.body.push(Instruction::Block(Box::new(else_block)));

                        // Exclude helper blocks from all parent instruction lists
                        builder.exclude_index(then_block_idx);
                        builder.exclude_index(else_block_idx);

                        // Create the If instruction
                        let if_inst = IfInst {
                            block_type,
                            then_block: NodeIdx(then_block_idx as u32),
                            else_block: NodeIdx(else_block_idx as u32),
                            next: FuncLoc { block_id: NodeIdx(0), instr_idx: 0 }, // Not used
                        };
                        builder.body[block_start] = Instruction::If(Box::new(if_inst));
                    }
                    BlockKind::Block | BlockKind::Loop => {
                        let instrs = builder.collect_instrs(block_start, block_end);
                        let is_loop = matches!(kind, BlockKind::Loop);
                        let block = BlockInst {
                            block_type,
                            instrs,
                            is_loop,
                            next: if builder.block_stack.is_empty() { None } else { Some(next_loc) },
                        };
                        builder.body[block_start] = Instruction::Block(Box::new(block));
                    }
                }
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

        WI::BrTable(bt) => {
            let labels: Vec<NodeIdx> = bt.labels.iter().map(|idx| {
                let depth = resolve_index(idx);
                if builder.block_stack.len() > depth as usize {
                    let stack_idx = builder.block_stack.len() - 1 - depth as usize;
                    NodeIdx(builder.block_stack[stack_idx].0 as u32)
                } else {
                    NodeIdx(0)
                }
            }).collect();

            let default_depth = resolve_index(&bt.default);
            let default = if builder.block_stack.len() > default_depth as usize {
                let stack_idx = builder.block_stack.len() - 1 - default_depth as usize;
                NodeIdx(builder.block_stack[stack_idx].0 as u32)
            } else {
                NodeIdx(0)
            };

            builder.emit(Instruction::BrTable { labels, default });
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

        // Truncation: float -> int
        WI::I32TruncF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F32, to: NumberType::I32 }); }
        WI::I32TruncF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F32, to: NumberType::I32 }); }
        WI::I32TruncF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F64, to: NumberType::I32 }); }
        WI::I32TruncF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F64, to: NumberType::I32 }); }
        WI::I64TruncF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F32, to: NumberType::I64 }); }
        WI::I64TruncF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F32, to: NumberType::I64 }); }
        WI::I64TruncF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncS, from: NumberType::F64, to: NumberType::I64 }); }
        WI::I64TruncF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncU, from: NumberType::F64, to: NumberType::I64 }); }

        // Saturating truncation: float -> int (no trap on overflow)
        WI::I32TruncSatF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F32, to: NumberType::I32 }); }
        WI::I32TruncSatF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F32, to: NumberType::I32 }); }
        WI::I32TruncSatF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F64, to: NumberType::I32 }); }
        WI::I32TruncSatF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F64, to: NumberType::I32 }); }
        WI::I64TruncSatF32S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F32, to: NumberType::I64 }); }
        WI::I64TruncSatF32U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F32, to: NumberType::I64 }); }
        WI::I64TruncSatF64S => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatS, from: NumberType::F64, to: NumberType::I64 }); }
        WI::I64TruncSatF64U => { builder.emit(Instruction::Convert { op: ConvertOp::TruncSatU, from: NumberType::F64, to: NumberType::I64 }); }

        // Conversion: int -> float
        WI::F32ConvertI32S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I32, to: NumberType::F32 }); }
        WI::F32ConvertI32U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I32, to: NumberType::F32 }); }
        WI::F32ConvertI64S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I64, to: NumberType::F32 }); }
        WI::F32ConvertI64U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I64, to: NumberType::F32 }); }
        WI::F64ConvertI32S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I32, to: NumberType::F64 }); }
        WI::F64ConvertI32U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I32, to: NumberType::F64 }); }
        WI::F64ConvertI64S => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertS, from: NumberType::I64, to: NumberType::F64 }); }
        WI::F64ConvertI64U => { builder.emit(Instruction::Convert { op: ConvertOp::ConvertU, from: NumberType::I64, to: NumberType::F64 }); }

        // Demote/Promote: float -> float
        WI::F32DemoteF64 => { builder.emit(Instruction::Convert { op: ConvertOp::Demote, from: NumberType::F64, to: NumberType::F32 }); }
        WI::F64PromoteF32 => { builder.emit(Instruction::Convert { op: ConvertOp::Promote, from: NumberType::F32, to: NumberType::F64 }); }

        // Reinterpret: same bits, different type
        WI::I32ReinterpretF32 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::F32, to: NumberType::I32 }); }
        WI::I64ReinterpretF64 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::F64, to: NumberType::I64 }); }
        WI::F32ReinterpretI32 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::I32, to: NumberType::F32 }); }
        WI::F64ReinterpretI64 => { builder.emit(Instruction::Convert { op: ConvertOp::Reinterpret, from: NumberType::I64, to: NumberType::F64 }); }

        // f32 operations
        WI::F32Const(v) => { builder.emit(Instruction::Const(Value::F32(f32::from_bits(v.bits)))); }
        WI::F32Add => { builder.emit(Instruction::Binary { op: BinaryOp::Add, typ: NumberType::F32 }); }
        WI::F32Sub => { builder.emit(Instruction::Binary { op: BinaryOp::Sub, typ: NumberType::F32 }); }
        WI::F32Mul => { builder.emit(Instruction::Binary { op: BinaryOp::Mul, typ: NumberType::F32 }); }
        WI::F32Div => { builder.emit(Instruction::Binary { op: BinaryOp::Div, typ: NumberType::F32 }); }
        WI::F32Neg => { builder.emit(Instruction::Unary { op: UnaryOp::Neg, typ: NumberType::F32 }); }
        WI::F32Abs => { builder.emit(Instruction::Unary { op: UnaryOp::Abs, typ: NumberType::F32 }); }
        WI::F32Sqrt => { builder.emit(Instruction::Unary { op: UnaryOp::Sqrt, typ: NumberType::F32 }); }
        WI::F32Ceil => { builder.emit(Instruction::Unary { op: UnaryOp::Ceil, typ: NumberType::F32 }); }
        WI::F32Floor => { builder.emit(Instruction::Unary { op: UnaryOp::Floor, typ: NumberType::F32 }); }
        WI::F32Trunc => { builder.emit(Instruction::Unary { op: UnaryOp::Trunc, typ: NumberType::F32 }); }
        WI::F32Nearest => { builder.emit(Instruction::Unary { op: UnaryOp::Nearest, typ: NumberType::F32 }); }
        WI::F32Min => { builder.emit(Instruction::Binary { op: BinaryOp::Min, typ: NumberType::F32 }); }
        WI::F32Max => { builder.emit(Instruction::Binary { op: BinaryOp::Max, typ: NumberType::F32 }); }
        WI::F32Copysign => { builder.emit(Instruction::Binary { op: BinaryOp::Copysign, typ: NumberType::F32 }); }
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
        WI::F64Sqrt => { builder.emit(Instruction::Unary { op: UnaryOp::Sqrt, typ: NumberType::F64 }); }
        WI::F64Ceil => { builder.emit(Instruction::Unary { op: UnaryOp::Ceil, typ: NumberType::F64 }); }
        WI::F64Floor => { builder.emit(Instruction::Unary { op: UnaryOp::Floor, typ: NumberType::F64 }); }
        WI::F64Trunc => { builder.emit(Instruction::Unary { op: UnaryOp::Trunc, typ: NumberType::F64 }); }
        WI::F64Nearest => { builder.emit(Instruction::Unary { op: UnaryOp::Nearest, typ: NumberType::F64 }); }
        WI::F64Min => { builder.emit(Instruction::Binary { op: BinaryOp::Min, typ: NumberType::F64 }); }
        WI::F64Max => { builder.emit(Instruction::Binary { op: BinaryOp::Max, typ: NumberType::F64 }); }
        WI::F64Copysign => { builder.emit(Instruction::Binary { op: BinaryOp::Copysign, typ: NumberType::F64 }); }
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
    while let Some((block_start, kind, block_type)) = builder.block_stack.pop() {
        let block_end = builder.current_idx();
        let instrs = builder.collect_instrs(block_start, block_end);
        let is_loop = matches!(kind, BlockKind::Loop);

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
            // Block params/results are just type annotations - params are already on stack,
            // results are what's left on stack. We don't validate, so just record the type.
            if ft.results.is_empty() {
                Ok(BlockType::None)
            } else if ft.results.len() == 1 {
                Ok(BlockType::Value(convert_val_type(&ft.results[0])?))
            } else {
                // Multi-value: just use None for now (we don't validate)
                Ok(BlockType::None)
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
        wast::core::WastArgCore::RefExtern(v) => Some(Value::Ref(*v as u64)),
        wast::core::WastArgCore::RefNull(_) => Some(Value::Ref(0)),
        _ => None,
    }
}

fn convert_core_ret(arg: &wast::core::WastRetCore) -> Option<Value> {
    match arg {
        wast::core::WastRetCore::I32(v) => Some(Value::I32(*v)),
        wast::core::WastRetCore::I64(v) => Some(Value::I64(*v)),
        wast::core::WastRetCore::F32(v) => match v {
            wast::core::NanPattern::Value(f) => Some(Value::F32(f32::from_bits(f.bits))),
            wast::core::NanPattern::CanonicalNan => Some(Value::F32(f32::NAN)),
            wast::core::NanPattern::ArithmeticNan => Some(Value::F32(f32::NAN)),
        },
        wast::core::WastRetCore::F64(v) => match v {
            wast::core::NanPattern::Value(f) => Some(Value::F64(f64::from_bits(f.bits))),
            wast::core::NanPattern::CanonicalNan => Some(Value::F64(f64::NAN)),
            wast::core::NanPattern::ArithmeticNan => Some(Value::F64(f64::NAN)),
        },
        wast::core::WastRetCore::RefExtern(v) => Some(Value::Ref(v.unwrap_or(0) as u64)),
        wast::core::WastRetCore::RefNull(_) => Some(Value::Ref(0)),
        _ => None,
    }
}
