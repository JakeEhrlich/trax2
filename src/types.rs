use std::cell::RefCell;
use std::rc::Rc;

// ============================================================================
// WASM Type System
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberType {
    I32,
    I64,
    F32,
    F64,
}

impl NumberType {
    pub fn size(&self) -> usize {
        match self {
            NumberType::I32 => 4,
            NumberType::I64 => 8,
            NumberType::F32 => 4,
            NumberType::F64 => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeapType {
    Any,
    Eq,
    I31,
    Struct,
    None,
    Func,
    NoFunc,
    Exn,
    NoExn,
    Extern,
    NoExtern,
    TypeUse(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefType {
    pub heap_type: HeapType,
    pub nullable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueType {
    V128,
    Number(NumberType),
    Ref(RefType),
}

impl ValueType {
    pub fn size(&self) -> usize {
        match self {
            ValueType::V128 => 16,
            ValueType::Number(num_type) => num_type.size(),
            ValueType::Ref(_) => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResultType {
    pub types: Vec<ValueType>,
}

#[derive(Clone, Debug)]
pub enum BlockType {
    None,
    Value(ValueType),
    TypeIdx(u32),
}

#[derive(Clone, Debug)]
pub enum CompositeType {
    Func(ResultType, ResultType),
}

#[derive(Clone, Debug)]
pub struct SubType {
    pub composite_type: CompositeType,
}

#[derive(Clone, Debug)]
pub struct RecType {
    pub subtypes: Vec<SubType>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AddrType {
    I32,
    I64,
}

#[derive(Clone, Debug)]
pub struct MemoryType {
    pub addr_type: AddrType,
    pub min: u64,
    pub max: Option<u64>,
}

// ============================================================================
// Code Location Types
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeIdx(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuncLoc {
    pub block_id: NodeIdx,
    pub instr_idx: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeLocation {
    pub func_id: u32,
    pub loc: FuncLoc,
}

// ============================================================================
// Instructions
// ============================================================================

#[derive(Clone, Debug)]
pub struct BlockInst {
    pub block_type: BlockType,
    pub instrs: Vec<NodeIdx>,
    pub is_loop: bool,
    /// Where to go after this block completes. None means implicit return
    /// (i.e., this is the function's entry block).
    pub next: Option<FuncLoc>,
}

impl BlockInst {
    /// Get the next location after executing an instruction at `loc`.
    /// Returns None if we've finished the block and there's no next location
    /// (meaning we should implicitly return from the function).
    pub fn next_loc(&self, loc: FuncLoc) -> Option<FuncLoc> {
        if loc.instr_idx as usize >= self.instrs.len() {
            return self.next;
        }
        Some(FuncLoc {
            block_id: loc.block_id,
            instr_idx: loc.instr_idx + 1,
        })
    }
}

#[derive(Clone, Debug)]
pub struct IfInst {
    pub block_type: BlockType,
    pub then_block: NodeIdx,
    pub else_block: NodeIdx,
    pub next: FuncLoc,
}

#[derive(Clone, Debug)]
pub struct MemArg {
    pub mem: u32,
    pub align: u32,
    pub offset: u64,
}

// ============================================================================
// Unified Numeric Operations
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Eqz,
    Neg,
    Abs,
    Clz,
    Ctz,
    Popcnt,
    Extend8S,
    Extend16S,
    Extend32S, // i64 only
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    Div,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    LtS,
    LtU,
    Lt,
    GtS,
    GtU,
    Gt,
    LeS,
    LeU,
    Le,
    GeS,
    GeU,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertOp {
    Wrap,
    ExtendS,
    ExtendU,
    TruncS,
    TruncU,
    Convert,
    Demote,
    Promote,
    Reinterpret,
}

#[derive(Clone, Debug)]
pub enum Instruction {
    Nop,
    Unreachable,
    Drop,
    Select(Option<ValueType>),
    Block(Box<BlockInst>),
    If(Box<IfInst>),
    Br(NodeIdx),
    BrIf(NodeIdx),
    Call(u32),
    Return,
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    TableGet(u32),
    TableSize(u32),
    Load { mem_arg: MemArg, val_type: NumberType },
    Store { mem_arg: MemArg, val_type: NumberType },
    MemorySize(u32),
    Const(Value),
    Unary { op: UnaryOp, typ: NumberType },
    Binary { op: BinaryOp, typ: NumberType },
    Compare { op: CompareOp, typ: NumberType },
    Convert { op: ConvertOp, from: NumberType, to: NumberType },
}

// ============================================================================
// Module Structure
// ============================================================================

#[derive(Clone, Debug)]
pub struct Function {
    pub type_idx: u32,
    pub locals: Vec<ValueType>,
    pub body: Vec<Instruction>,
}

#[derive(Clone, Debug)]
pub struct Module {
    pub types: Vec<RecType>,
    pub funcs: Vec<Function>,
    pub mems: Vec<MemoryType>,
    pub start: Option<u32>,
}

// ============================================================================
// Runtime Values
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    V128([u8; 16]),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ref(u64),
}

impl Value {
    pub fn from_bytes(val_type: &ValueType, data: &[u8]) -> Option<Value> {
        Some(match val_type {
            ValueType::V128 => {
                let arr: [u8; 16] = data.try_into().ok()?;
                Value::V128(arr)
            }
            ValueType::Ref(_) => {
                let arr: [u8; 8] = data.try_into().ok()?;
                Value::Ref(u64::from_le_bytes(arr))
            }
            ValueType::Number(NumberType::I64) => {
                let arr: [u8; 8] = data.try_into().ok()?;
                Value::I64(i64::from_le_bytes(arr))
            }
            ValueType::Number(NumberType::I32) => {
                let arr: [u8; 4] = data.try_into().ok()?;
                Value::I32(i32::from_le_bytes(arr))
            }
            ValueType::Number(NumberType::F64) => {
                let arr: [u8; 8] = data.try_into().ok()?;
                Value::F64(f64::from_le_bytes(arr))
            }
            ValueType::Number(NumberType::F32) => {
                let arr: [u8; 4] = data.try_into().ok()?;
                Value::F32(f32::from_le_bytes(arr))
            }
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Value::V128(x) => x,
            Value::I32(x) => unsafe {
                std::slice::from_raw_parts(x as *const i32 as *const u8, 4)
            },
            Value::I64(x) => unsafe {
                std::slice::from_raw_parts(x as *const i64 as *const u8, 8)
            },
            Value::F32(x) => unsafe {
                std::slice::from_raw_parts(x as *const f32 as *const u8, 4)
            },
            Value::F64(x) => unsafe {
                std::slice::from_raw_parts(x as *const f64 as *const u8, 8)
            },
            Value::Ref(x) => unsafe {
                std::slice::from_raw_parts(x as *const u64 as *const u8, 8)
            },
        }
    }

    pub fn value_type(&self) -> ValueType {
        match self {
            Value::V128(_) => ValueType::V128,
            Value::I32(_) => ValueType::Number(NumberType::I32),
            Value::I64(_) => ValueType::Number(NumberType::I64),
            Value::F32(_) => ValueType::Number(NumberType::F32),
            Value::F64(_) => ValueType::Number(NumberType::F64),
            Value::Ref(_) => ValueType::Ref(RefType {
                heap_type: HeapType::Any,
                nullable: true,
            }),
        }
    }

    pub fn number_type(&self) -> Option<NumberType> {
        match self {
            Value::I32(_) => Some(NumberType::I32),
            Value::I64(_) => Some(NumberType::I64),
            Value::F32(_) => Some(NumberType::F32),
            Value::F64(_) => Some(NumberType::F64),
            _ => None,
        }
    }

    pub fn is_falsey(&self) -> Option<bool> {
        match self {
            Value::I32(x) => Some(*x == 0),
            Value::I64(x) => Some(*x == 0),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> Option<bool> {
        self.is_falsey().map(|x| !x)
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self { Value::I32(x) => Some(*x), _ => None }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self { Value::I64(x) => Some(*x), _ => None }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self { Value::F32(x) => Some(*x), _ => None }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self { Value::F64(x) => Some(*x), _ => None }
    }

    pub fn default_for(val_type: &ValueType) -> Value {
        match val_type {
            ValueType::V128 => Value::V128([0; 16]),
            ValueType::Number(NumberType::I32) => Value::I32(0),
            ValueType::Number(NumberType::I64) => Value::I64(0),
            ValueType::Number(NumberType::F32) => Value::F32(0.0),
            ValueType::Number(NumberType::F64) => Value::F64(0.0),
            ValueType::Ref(_) => Value::Ref(0),
        }
    }
}

// ============================================================================
// Memory and MemoryView
// ============================================================================

pub const PAGE_SIZE: usize = 65536;

/// Owns the underlying byte storage. Allocated in pages.
/// Clone shares the same backing data (shared mutable reference).
#[derive(Debug, Clone)]
pub struct Memory {
    data: Rc<RefCell<Vec<u8>>>,
}

impl Memory {
    pub fn new(pages: u32) -> Self {
        let size = (pages as usize) * PAGE_SIZE;
        Memory {
            data: Rc::new(RefCell::new(vec![0u8; size])),
        }
    }

    pub fn capacity(&self) -> usize {
        self.data.borrow().len()
    }

    pub fn pages(&self) -> u32 {
        (self.capacity() / PAGE_SIZE) as u32
    }

    pub fn view(&self, start: usize, end: usize) -> Option<MemoryView> {
        if end > self.capacity() || start > end {
            return None;
        }
        Some(MemoryView {
            data: Rc::clone(&self.data),
            start,
            end,
            capacity: end,
        })
    }

    pub fn full_view(&self) -> MemoryView {
        let cap = self.capacity();
        MemoryView {
            data: Rc::clone(&self.data),
            start: 0,
            end: cap,
            capacity: cap,
        }
    }
}

/// A mutable view into Memory with byte-level boundaries.
/// start/end can be mutated to support stack-like growth.
/// capacity tracks the maximum end value for this view.
#[derive(Debug, Clone)]
pub struct MemoryView {
    data: Rc<RefCell<Vec<u8>>>,
    pub start: usize,
    pub end: usize,
    pub capacity: usize,
}

impl MemoryView {
    pub fn size(&self) -> usize {
        self.end - self.start
    }

    pub fn set_bytes(&self, offset: usize, bytes: &[u8]) -> bool {
        let abs_start = self.start + offset;
        let abs_end = abs_start + bytes.len();
        if abs_end > self.end {
            return false;
        }
        let mut data = self.data.borrow_mut();
        data[abs_start..abs_end].copy_from_slice(bytes);
        true
    }

    pub fn get_value(&self, offset: usize, val_type: &ValueType) -> Option<Value> {
        let abs_start = self.start + offset;
        let abs_end = abs_start + val_type.size();
        if abs_end > self.end {
            return None;
        }
        let data = self.data.borrow();
        Value::from_bytes(val_type, &data[abs_start..abs_end])
    }

    pub fn set_value(&self, offset: usize, value: &Value) -> bool {
        self.set_bytes(offset, value.as_bytes())
    }

    pub fn subview(&self, rel_start: usize, rel_end: usize) -> Option<MemoryView> {
        let abs_start = self.start + rel_start;
        let abs_end = self.start + rel_end;
        if abs_end > self.end || abs_start > abs_end {
            return None;
        }
        Some(MemoryView {
            data: Rc::clone(&self.data),
            start: abs_start,
            end: abs_end,
            capacity: abs_end,
        })
    }

    pub fn grow_end(&mut self, amount: usize) -> bool {
        let new_end = self.end + amount;
        if new_end > self.capacity {
            return false;
        }
        self.end = new_end;
        true
    }

    pub fn shrink_end(&mut self, amount: usize) -> bool {
        if amount > self.size() {
            return false;
        }
        self.end -= amount;
        true
    }

    pub fn grow_start(&mut self, amount: usize) -> bool {
        if amount > self.start {
            return false;
        }
        self.start -= amount;
        true
    }

    pub fn shrink_start(&mut self, amount: usize) -> bool {
        if amount > self.size() {
            return false;
        }
        self.start += amount;
        true
    }

    pub fn size_pages(&self) -> u32 {
        (self.size() / PAGE_SIZE) as u32
    }
}

// ============================================================================
// Typed Stack (backed by MemoryView, manipulates view bounds)
// ============================================================================

#[derive(Debug, Clone)]
pub struct TypedStack {
    pub view: MemoryView,
    types: Vec<(usize, ValueType)>,
}

impl TypedStack {
    pub fn new(mut view: MemoryView) -> Self {
        let start = view.start;
        view.end = start;
        TypedStack { view, types: Vec::new() }
    }

    pub fn push(&mut self, value: Value) -> bool {
        let val_type = value.value_type();
        let size = val_type.size();
        let offset = self.view.size();

        if !self.view.grow_end(size) {
            return false;
        }
        if !self.view.set_value(offset, &value) {
            self.view.shrink_end(size);
            return false;
        }
        self.types.push((offset, val_type));
        true
    }

    pub fn pop(&mut self) -> Option<Value> {
        let (offset, val_type) = self.types.pop()?;
        let value = self.view.get_value(offset, &val_type)?;
        self.view.shrink_end(val_type.size());
        Some(value)
    }

    pub fn peek(&self) -> Option<Value> {
        let (offset, ref val_type) = self.types.last()?;
        self.view.get_value(*offset, val_type)
    }

    pub fn len(&self) -> usize { self.types.len() }
    pub fn is_empty(&self) -> bool { self.types.is_empty() }
    pub fn byte_size(&self) -> usize { self.view.size() }
}

// ============================================================================
// Typed Locals (backed by MemoryView)
// ============================================================================

#[derive(Debug, Clone)]
pub struct TypedLocals {
    view: MemoryView,
    layout: Vec<(usize, ValueType)>,
}

impl TypedLocals {
    pub fn new(view: MemoryView, types: &[ValueType]) -> Option<Self> {
        let mut offset = 0usize;
        let mut layout = Vec::with_capacity(types.len());

        for val_type in types {
            let size = val_type.size();
            if offset + size > view.size() {
                return None;
            }
            let zero = Value::default_for(val_type);
            view.set_value(offset, &zero);
            layout.push((offset, val_type.clone()));
            offset += size;
        }

        Some(TypedLocals { view, layout })
    }

    pub fn get(&self, idx: u32) -> Option<Value> {
        let (offset, ref val_type) = self.layout.get(idx as usize)?;
        self.view.get_value(*offset, val_type)
    }

    pub fn set(&self, idx: u32, value: Value) -> bool {
        let Some((offset, ref expected_type)) = self.layout.get(idx as usize) else {
            return false;
        };
        if value.value_type() != *expected_type {
            return false;
        }
        self.view.set_value(*offset, &value)
    }

    pub fn len(&self) -> usize { self.layout.len() }
    pub fn is_empty(&self) -> bool { self.layout.is_empty() }

    pub fn byte_size(&self) -> usize {
        self.layout.last().map(|(off, t)| off + t.size()).unwrap_or(0)
    }
}

// ============================================================================
// Frame and Continuation
// ============================================================================

#[derive(Debug, Clone)]
pub struct Frame {
    pub loc: CodeLocation,
    pub locals: TypedLocals,
    pub stack: TypedStack,
}

#[derive(Debug, Clone)]
pub struct Continuation {
    pub frame: Frame,
    pub parent: Option<Box<Continuation>>,
}

impl Continuation {
    pub fn new(loc: CodeLocation, locals: TypedLocals, stack: TypedStack) -> Self {
        Continuation {
            frame: Frame { loc, locals, stack },
            parent: None,
        }
    }

    pub fn with_parent(
        loc: CodeLocation,
        locals: TypedLocals,
        stack: TypedStack,
        parent: Continuation,
    ) -> Self {
        Continuation {
            frame: Frame { loc, locals, stack },
            parent: Some(Box::new(parent)),
        }
    }
}
