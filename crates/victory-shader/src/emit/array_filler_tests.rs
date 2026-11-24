//! A descriptor dword the console compiler only puts into an element of an
//! M0-indexed VGPR array (lines9686-rgb 261b653). Of a buffer descriptor
//! this tree gives the host view's address (`Lower::view_address`) before
//! that rule is asked, so these programs translate either way; the rule
//! (0 for a dword with no host value) is for image descriptors.

use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A compute program of 32 threads: the UAV's descriptor from the table in
/// s[4:7], `body`, then v2 stored at the lane's dword.
fn program(body: &[u32]) -> NativeProgram {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ]);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn emit(native: &NativeProgram) -> Result<super::HostShader, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let container = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&container)?;
    super::emit(&c, native, &p, &st, &io, &res)
}

/// v4..v7 indexed through M0, element 1 filled with the UAV descriptor's
/// address dword (s4); lane t reads element 0, its own id.
const ARRAY_FILLER: &[u32] = &[
    0x7E08_0300, // v_mov_b32 v4, v0
    0x7E0A_0204, // v_mov_b32 v5, s4
    0x7E0C_0280, // v_mov_b32 v6, 0
    0x7E0E_0280, // v_mov_b32 v7, 0
    0xBEFC_0380, // s_mov_b32 m0, 0
    0x7E04_8704, // v_movrels_b32 v2, v4
];

/// The same array rewritten first through `v_movreld`.
const ARRAY_FILLER_REWRITTEN: &[u32] = &[
    0x7E08_0280, // v_mov_b32 v4, 0
    0x7E0A_0204, // v_mov_b32 v5, s4
    0x7E0C_0280, // v_mov_b32 v6, 0
    0x7E0E_0280, // v_mov_b32 v7, 0
    0xBEFC_0381, // s_mov_b32 m0, 1
    0x7E08_8500, // v_movreld_b32 v4, v0
    0x7E04_8704, // v_movrels_b32 v2, v4
];

/// The address dword copied, then used as a number.
const ADDRESS_AS_NUMBER: &[u32] = &[
    0x7E0A_0204, // v_mov_b32 v5, s4
    0x4A04_0105, // v_add_nc_u32 v2, v5, v0
];

/// An indexed array beside it, the address dword copied below the array
/// and added to its result.
const ADDRESS_BESIDE_AN_ARRAY: &[u32] = &[
    0x7E08_0300, // v_mov_b32 v4, v0
    0x7E0A_0280, // v_mov_b32 v5, 0
    0x7E0C_0280, // v_mov_b32 v6, 0
    0x7E0E_0280, // v_mov_b32 v7, 0
    0x7E06_0204, // v_mov_b32 v3, s4
    0xBEFC_0380, // s_mov_b32 m0, 0
    0x7E04_8704, // v_movrels_b32 v2, v4
    0x4A04_0503, // v_add_nc_u32 v2, v3, v2
];

#[test]
fn a_buffer_descriptor_dword_in_an_indexed_array_translates() {
    for body in [ARRAY_FILLER, ARRAY_FILLER_REWRITTEN, ADDRESS_AS_NUMBER, ADDRESS_BESIDE_AN_ARRAY] {
        if let Err(e) = emit(&program(body)) {
            panic!("{e}");
        }
    }
}
