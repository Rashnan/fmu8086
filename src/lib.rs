pub mod fmu8086 {

}

pub mod ias {
    use std::{ops::{Add, Neg, Sub, Mul, Div}, cmp};

    // NOTE: tried to force everything to use exact bits wherever possible
    // NOTE: tried not to take shortcuts

    // supports only integer types lecture 4 slide 27
    #[derive(Debug, Clone)]
    pub struct IasNumber {
        pub sign: bool,// is negative?
        pub number: [bool; IasNumber::BITS] // contains bits 1 to 39
    }

    impl IasNumber {
        pub const BITS: usize = 39;
        pub const MAX: IasNumber = IasNumber { sign: false, number: [true; IasNumber::BITS] };
        pub const MIN: IasNumber = IasNumber { sign: true, number: [false; IasNumber::BITS] };
        pub const ZERO: IasNumber = IasNumber { sign: false, number: [false; IasNumber::BITS] };

        pub const fn zero() -> Self {
            Self { sign: false, number: [false; IasNumber::BITS] }
        }
        pub fn from_i64(num: i64) -> Self {
            // max range is (2^39 - 1) - (-2^39)
            assert!(num < 2i64.pow(IasNumber::BITS as u32) && num >= -2i64.pow(IasNumber::BITS as u32));

            let mut number = [false; IasNumber::BITS];
            number.iter_mut()
                .rev()
                .enumerate()
                .for_each(|(i, x)| {
                    *x = (num & (1 << i)) != 0;
                });

            Self { 
                sign: num < 0,
                number,
            }
        }

        pub fn to_i64(&self) -> i64 {
            let num = self.number.iter()
                .fold(0i64, |acc, x| (acc << 1) | (*x as u64 as i64));

            if self.sign {
                (-1i64 << IasNumber::BITS) + num
            }
            else {
                num
            }
        }
    }

    impl PartialEq for IasNumber {
        fn eq(&self, other: &Self) -> bool {
            self.to_i64().eq(&other.to_i64())
        }
    }
    impl Eq for IasNumber {}
    impl PartialOrd for IasNumber {
        fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
            self.to_i64().partial_cmp(&other.to_i64())
        }
    }
    impl Ord for IasNumber {
        fn cmp(&self, other: &Self) -> cmp::Ordering {
            self.to_i64().cmp(&other.to_i64())
        }
    }
    impl Add for IasNumber {
        type Output = IasNumber;
        fn add(self, rhs: Self) -> Self::Output {
            Self::from_i64(self.to_i64() + rhs.to_i64())
        }
    }
    impl Neg for IasNumber {
        type Output = IasNumber;
        fn neg(self) -> Self::Output {
            Self::Output { 
                sign: !self.sign,
                number: self.number
            }
        }
    }
    impl Sub for IasNumber {
        type Output = IasNumber;
        fn sub(self, rhs: Self) -> Self::Output {
            Self::from_i64(self.to_i64() - rhs.to_i64())
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct IasNumberMulResult {
        pub msbs: IasNumber,
        pub lsbs: IasNumber,
    }
    impl Mul for IasNumber {
        type Output = IasNumberMulResult;
        fn mul(self, rhs: Self) -> Self::Output {
            // 39b * 39bb = 78b > 64b
            let res: i128 = self.to_i64() as i128 * rhs.to_i64() as i128;
            Self::Output {
                msbs: IasNumber::from_i64((IasNumber::MAX.to_i64() << IasNumber::BITS) & res as i64),
                lsbs: IasNumber::from_i64(IasNumber::MAX.to_i64() & res as i64),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct IasNumberDivResult {
        pub quotient: IasNumber,
        pub remainder: IasNumber,
    }
    impl Div for IasNumber {
        type Output = IasNumberDivResult;
        fn div(self, rhs: Self) -> Self::Output {
            Self::Output {
                quotient: IasNumber::from_i64(self.to_i64() / rhs.to_i64()),
                remainder: IasNumber::from_i64(self.to_i64() % rhs.to_i64()),
            }
        }
    }
    
    #[repr(u8)]
    #[derive(Debug, Clone)]
    pub enum IasOpcodesEnum {
        Halt,// NOTE: NOT IN SPEC but https://cs.colby.edu/djskrien/IASSim/index.html says a halt instruction is needed
        
        Load,// load mem into ac
        LoadNeg,// load -mem into ac
        LoadAbs,// load |mem| into ac
        LoadNegAbs,// load -|mem| into ac
        Add,// add mem into ac
        Sub,// sub mem from ac and puts into ac
        AddAbs,// add |mem| into ac
        SubAbs,// sub |mem| from ac and puts into ac

        LoadMqMem,// load into mq
        LoadMq,// load mq into ac
        Mul,// mul mem by mq and MSBs go to ac and LSBs stay in mq
        Div,// div ac by mq, quotient goes into mq and remainder into ac
        
        JumpLeft,// jump to instruction on left half of memory
        JumpRight,// jump to instruction on right half of 
        JumpLeftIfNotNeg,// jump left if ac is non-neg
        JumpRightIfNotNeg,// jump right if ac is non-neg
        
        Stor,// transfer ac to mem
        StorLeftAddr,// replace left address in mem by 12 LSBs of ac
        StorRightAddr,// replace right address in mem by 12 LSBs of ac
        
        Lsh,// left shift bits (zero fill)
        Rsh,// right shift bits (sign extend)
    }

    impl IasOpcodesEnum {
        pub const BITS: usize = 8;
    }

    // CRYING
    impl From<u8> for IasOpcodesEnum {
        fn from(value: u8) -> Self {
            match value {
                0 => Self::Halt,
                
                1 => Self::Load,
                2 => Self::LoadNeg,
                3 => Self::LoadAbs,
                4 => Self::LoadNegAbs,
                5 => Self::Add,
                6 => Self::Sub,
                7 => Self::AddAbs,
                8 => Self::SubAbs,
                
                9 => Self::LoadMqMem,
                10 => Self::LoadMq,
                11 => Self::Mul,
                12 => Self::Div,
                
                13 => Self::JumpLeft,
                14 => Self::JumpRight,
                15 => Self::JumpLeftIfNotNeg,
                16 => Self::JumpRightIfNotNeg,
                
                17 => Self::Stor,
                18 => Self::StorLeftAddr,
                19 => Self::StorRightAddr,
                
                20 => Self::Lsh,
                21 => Self::Rsh,
                
                _ => Self::Halt,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct IasAddress {
        pub data: [bool; IasAddress::BITS],
    }
    
    impl IasAddress {
        pub const BITS: usize = 12;
    
        pub const fn zero() -> Self {
            Self { data: [false; IasAddress::BITS] }
        }
        pub fn from_u16(num: u16) -> Self {
            // max range is (2^39 - 1) - (-2^39)
            assert!(num < 2u16.pow(IasAddress::BITS as u32 + 1));

            let mut data = [false; IasAddress::BITS];
            data.iter_mut()
                .rev()
                .enumerate()
                .for_each(|(i, x)| {
                    *x = (num & (1 << i)) != 0;
                });

            Self { data }
        }
        
        pub fn to_u16(&self) -> u16 {
            self.data.iter()
                .fold(0u16, |acc, x| (acc << 1) | (*x as u16))
        }
    }

    #[derive(Debug, Clone)]
    pub struct IasInstruction {
        pub opcode: IasOpcodesEnum,
        pub address: IasAddress, 
    }

    impl IasInstruction {
        pub fn new() -> Self {
            Self {
                opcode: IasOpcodesEnum::Halt,
                address: IasAddress::zero(),
            }
        }

        pub fn run(&self, machine: &mut IasMachine, memory: &mut Vec<IasWord>) {
            machine.ir = self.address.clone();

            match &self.opcode {
                IasOpcodesEnum::Halt => {
                    machine.halt = true;
                },

                IasOpcodesEnum::Load => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = IasNumber::from_i64(
                        {
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        }
                    );
                },
                IasOpcodesEnum::LoadNeg => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = IasNumber::from_i64(
                        -{
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        }
                    );
                },
                IasOpcodesEnum::LoadAbs => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = IasNumber::from_i64(
                        i64::abs({
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        })
                    );
                },
                IasOpcodesEnum::LoadNegAbs => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = IasNumber::from_i64(
                        -i64::abs({
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        })
                    );
                },

                IasOpcodesEnum::Add => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = machine.ac.clone() + IasNumber::from_i64(
                        {
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        }
                    );
                },
                IasOpcodesEnum::Sub => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = machine.ac.clone() - IasNumber::from_i64(
                        {
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        }
                    );
                },
                IasOpcodesEnum::AddAbs => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = machine.ac.clone() + IasNumber::from_i64(
                        i64::abs({
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        })
                    );
                },
                IasOpcodesEnum::SubAbs => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.ac = machine.ac.clone() - IasNumber::from_i64(
                        i64::abs({
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        })
                    );
                },

                IasOpcodesEnum::LoadMqMem => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    machine.mq = IasNumber::from_i64(
                        {
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        }
                    );
                },
                IasOpcodesEnum::LoadMq => {
                    machine.ac = machine.mq.clone()
                },
                IasOpcodesEnum::Mul => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    let IasNumberMulResult { msbs, lsbs } = machine.mq.clone() * IasNumber::from_i64(
                        {
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        }
                    );
                    machine.ac = msbs;
                    machine.mq = lsbs;
                },
                IasOpcodesEnum::Div => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    let IasNumberDivResult { quotient, remainder } = machine.mq.clone() / IasNumber::from_i64(
                        {
                            match &machine.mdr {
                                // wont happen
                                IasWord::Instruction(ins) => ins.to_u64() as i64,
                                // always happens
                                IasWord::Number(number) => number.to_i64()
                            }
                        }
                    );
                    machine.ac = remainder;
                    machine.mq = quotient;
                },

                IasOpcodesEnum::JumpLeft => {
                    machine.pc = self.address.clone();
                    machine.offset_pc = true;
                },
                IasOpcodesEnum::JumpRight => {
                    machine.pc = self.address.clone();
                },
                IasOpcodesEnum::JumpLeftIfNotNeg => {
                    if machine.ac >= IasNumber::from_i64(0) {
                        machine.pc = self.address.clone();
                        machine.offset_pc = true;
                    }
                },
                IasOpcodesEnum::JumpRightIfNotNeg => {
                    if machine.ac >= IasNumber::from_i64(0) {
                        machine.pc = self.address.clone();
                    }
                },

                IasOpcodesEnum::Stor => {
                    machine.mar = self.address.clone();
                    machine.mdr = IasWord::Number(machine.ac.clone());
                    memory[machine.mar.to_u16() as usize] = machine.mdr.clone();
                },
                IasOpcodesEnum::StorLeftAddr => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    // modifies only bits 8 to 19 of memory location with 12 LSB of AC
                    match &mut machine.mdr {
                        IasWord::Instruction(ins) => {
                            ins.left.address = IasAddress::from_u16((machine.ac.to_i64() & 0x3f) as u16)
                        }
                        IasWord::Number(num) => {
                            let mut res = (machine.ac.to_i64() & 0x3f) as u16;
                            num.number.iter_mut()
                                .skip(IasOpcodesEnum::BITS - 1)
                                .take(IasAddress::BITS)
                                .rev()
                                .for_each(|x| {
                                    *x = res & 1 != 0;
                                    res >>= 1;
                                });
                        }
                    }
                    memory[machine.mar.to_u16() as usize] = machine.mdr.clone();
                },
                IasOpcodesEnum::StorRightAddr => {
                    machine.mar = self.address.clone();
                    machine.mdr = memory[machine.mar.to_u16() as usize].clone();
                    // modifies only bits 28 to 39 of memory location with 12 LSB of AC
                    match &mut machine.mdr {
                        IasWord::Instruction(ins) => {
                            ins.right.address = IasAddress::from_u16((machine.ac.to_i64() & 0x3f) as u16)
                        }
                        IasWord::Number(num) => {
                            let mut res = (machine.ac.to_i64() & 0x3f) as u16;
                            num.number.iter_mut()
                                .rev()
                                .take(IasAddress::BITS)
                                .for_each(|x| {
                                    *x = res & 1 != 0;
                                    res >>= 1;
                                });
                        }
                    }
                    memory[machine.mar.to_u16() as usize] = machine.mdr.clone();
                },

                IasOpcodesEnum::Lsh => {
                    let num = ((machine.ac.to_i64() << IasNumber::BITS) as i128 + machine.mq.to_i64() as i128) << 1;
                    machine.ac = IasNumber::from_i64((num >> IasNumber::BITS) as i64);
                    machine.mq = IasNumber::from_i64((num & IasNumber::MAX.to_i64() as i128) as i64);
                },
                IasOpcodesEnum::Rsh => {
                    let mut num = ((machine.ac.to_i64() << IasNumber::BITS) as i128 + machine.mq.to_i64() as i128) >> 1;
                    // ensure sign is extended
                    num |= (machine.ac.sign as i128) << (IasNumber::BITS * 2 - 1);
                    machine.ac = IasNumber::from_i64((num >> IasNumber::BITS) as i64);
                    machine.mq = IasNumber::from_i64((num & IasNumber::MAX.to_i64() as i128) as i64);
                },
            }
        }
        
        pub fn from_u32(num: u32) -> Self {
            // max range is (2^39 - 1) - (-2^39)
            assert!(num < 2u32.pow(IasAddress::BITS as u32 + 1));

            Self {
                opcode: IasOpcodesEnum::from((num >> 12) as u8),
                address: IasAddress::from_u16((num & (0x3f)) as u16)
            }
        }
        
        pub fn to_u32(&self) -> u32 {
            ((self.opcode.clone() as u8) << 12) as u32 + self.address.to_u16() as u32
        }
    
    }

    #[derive(Debug, Clone)]
    pub struct IasInstructionFull {
        pub left: IasInstruction,
        pub right: IasInstruction,
    }

    impl IasInstructionFull {
        pub fn new() -> Self {
            Self {
                left: IasInstruction::new(),
                right: IasInstruction::new()
            }
        }
        pub fn from_u64(num: u64) -> Self {
            Self {
                left: IasInstruction::from_u32((num >> 32) as u32),
                right: IasInstruction::from_u32((num & (0xffff)) as u32)
            }
        }

        pub fn to_u64(&self) -> u64 {
            (self.left.to_u32() << 32) as u64 + self.right.to_u32() as u64
        }
    }

    #[derive(Debug, Clone)]
    pub enum IasWord {
        Number(IasNumber),
        Instruction(IasInstructionFull),
    }

    // const MAX_MEMORY: usize = 4096usize;
    // pub struct IasMemory {
    //     // data: [IasInstruction; MAX_MEMORY]// 4096 * 40 ~ 160kb
    //     pub data: Vec<IasInstruction>
    // }

    pub struct IasMachine {
        // registers

        // data
        pub ac: IasNumber,
        pub mq: IasNumber,

        // technically stores both
        pub mdr: IasWord,

        // address
        pub pc: IasAddress,
        pub mar: IasAddress,
        pub ir: IasAddress,

        pub offset_pc: bool,

        pub halt: bool,
    }

    impl IasMachine {
        pub fn new() -> Self {
            Self {
                ac: IasNumber::zero(),
                mq: IasNumber::zero(),
                // stores temp data and temp 
                mdr: IasWord::Number(IasNumber::zero()),

                pc: IasAddress::zero(),
                mar: IasAddress::zero(),
                ir: IasAddress::zero(),
                offset_pc: false,

                halt: true,
            }
        }

        // IAS memory is stupider
        // so u cannot differentiate data
        // and instructions
        pub fn run(&mut self, mut memory: &mut Vec<IasWord>) {
            while !self.halt {
                // double clone because later wont allow me to return self mut
                // as mbr items also included in self
                // can be limited to only one clone but mbr is for visibility and debugging

                self.mar = IasAddress::from_u16(self.pc.to_u16());
                self.mdr = memory[self.mar.to_u16() as usize].clone();

                // increase pc
                self.pc = IasAddress::from_u16(self.pc.to_u16() + 1);

                match &self.mdr.clone() {
                    IasWord::Number(_number) => {
                        // ignore
                    },
                    IasWord::Instruction(IasInstructionFull { left, right }) => {
                        if !self.offset_pc {
                            // runs as long as not halted before
                            right.run(self, &mut memory);
                        }
                        if !self.halt {
                            left.run(self, &mut memory);
                            
                            // ensure offset pc is reset
                            self.offset_pc = false;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::ias::IasNumber;

    use super::*;

    // ias number

    #[test]
    fn ias_number_tests() {

    }
}