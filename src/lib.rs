#[macro_use]
extern crate clap;
#[macro_use]
extern crate derive_more;

use std::borrow::Cow;
use std::cmp::{max, min};
use std::collections::BTreeSet;
use std::iter::Iterator;
use std::path::PathBuf;

use cpp_demangle::*;
use fallible_iterator::FallibleIterator;
use gimli::{AttributeValue, CompilationUnitHeader, EndianSlice};
use rayon::prelude::*;
use regex::Regex;
use structopt::StructOpt;

trait Reader: gimli::Reader<Offset = usize> + Send + Sync {
    type SyncSendEndian: gimli::Endianity + Send + Sync;
}

impl<'input, Endian> Reader for gimli::EndianSlice<'input, Endian>
where
    Endian: gimli::Endianity + Send + Sync,
{
    type SyncSendEndian = Endian;
}

arg_enum! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    pub enum Language {
        Cpp,
        Rust,
    }
}

#[derive(StructOpt, Clone, Default)]
/// Evaluate the quality of debuginfo
#[structopt(name = "debuginfo-quality")]
pub struct Opt {
    /// Show results for each function. Print the worst functions first.
    #[structopt(short = "f", long = "functions")]
    pub functions: bool,
    /// Show results for each variable. Print the worst functions first.
    #[structopt(short = "v", long = "variables")]
    pub variables: bool,
    /// Treat locations with DW_OP_GNU_entry_value as missing in the main file
    #[structopt(long = "no-entry-value")]
    pub no_entry_value: bool,
    /// Treat locations with DW_OP_GNU_entry_value as missing in the baseline file
    #[structopt(long = "no-entry-value-baseline")]
    pub no_entry_value_baseline: bool,
    /// Treat locations with DW_OP_GNU_parameter_ref as missing in the main file
    #[structopt(long = "no-parameter-ref")]
    pub no_parameter_ref: bool,
    /// Treat locations with DW_OP_GNU_parameter_ref as missing in the baseline file
    #[structopt(long = "no-parameter-ref-baseline")]
    pub no_parameter_ref_baseline: bool,
    /// Only consider local variables
    #[structopt(long = "only-locals")]
    pub only_locals: bool,
    /// Only consider parameters
    #[structopt(long = "only-parameters")]
    pub only_parameters: bool,
    /// Regex to match function names against
    #[structopt(short = "s", long = "select-functions")]
    pub select_functions: Option<Regex>,
    /// Languages to look at
    #[structopt(
        short = "l",
        long = "language",
        raw(possible_values = "&Language::variants()", case_insensitive = "true")
    )]
    pub language: Option<Language>,
    /// Output using tab-separated values (TSV) format
    #[structopt(long = "tsv")]
    pub tsv: bool,
    /// File to use as a baseline. We try to match up functions in this file
    /// against functions in the main file; for all matches, subtract the scope coverage
    /// percentage in the baseline from the percentage of the main file.
    #[structopt(long = "baseline", parse(from_os_str))]
    pub baseline: Option<PathBuf>,
    /// Baseline defines the start of coverage in source line terms. We match all variables in this
    /// file to the main file. For each variable in the main file, the source line range is trimmed
    /// to start on the first source line used for that variable in this file.
    #[structopt(long = "range-start-baseline")]
    pub range_start_baseline: bool,
    /// Extend variable source line range from end of baseline to the end of parent scope. We match
    /// all variables in this file to the main file. For each variable in the main file, an
    /// additional coverage source line range is added from the point where baseline coverage ends
    /// to the end of parent scope. This simulates a debugging workflow that keeps variables live
    /// until the end of their scope.
    #[structopt(long = "extend-from-baseline")]
    pub extend_from_baseline: bool,
    /// File describing source line regions used to filter the covered lines in some way.
    #[structopt(long = "regions", parse(from_os_str))]
    pub regions: Option<PathBuf>,
    /// Report source-based scope lines from declaration regions.
    /// This will be further filtered by computation and definition regions if they are enabled.
    #[structopt(long = "scope-regions")]
    pub scope_regions: bool,
    /// Filters covered lines to only those within computation regions.
    #[structopt(long = "only-computation-regions")]
    pub only_computation_regions: bool,
    /// The first region where a given variable must be defined determines the start of the
    /// covered source line range.
    #[structopt(long = "range-start-first-defined-region")]
    pub range_start_first_defined_region: bool,
    /// File to analyze
    #[structopt(parse(from_os_str))]
    pub file: PathBuf,
}

#[derive(Clone, Default, Add, AddAssign)]
pub struct StatsBundle {
    pub parameters: VariableStats,
    pub variables: VariableStats,
}

#[derive(Clone)]
pub struct Stats {
    pub bundle: StatsBundle,
    pub opt: Opt,
    pub output: Vec<FunctionStats>,
}

#[derive(Clone)]
pub struct NamedVarStats {
    pub inlines: Vec<String>,
    pub name: String,
    pub decl_dir: String,
    pub decl_file: String,
    pub decl_line: String,
    pub extra: ExtraVarInfo,
    pub stats: VariableStats,
}

#[derive(Clone)]
pub struct ExtraVarInfo {
    pub source_line_set_covered: BTreeSet<u64>,
    pub source_line_set_after_covered: Option<BTreeSet<u64>>,
}

#[derive(Clone)]
pub struct FunctionStats {
    pub name: String,
    pub unit_dir: String,
    pub unit_name: String,
    pub stats: VariableStats,
    pub variables: Vec<NamedVarStats>,
}

struct UnitStats<'a> {
    bundle: StatsBundle,
    opt: &'a Opt,
    noninline_function_stack: Vec<Option<FunctionStats>>,
    output: Vec<FunctionStats>,
}

struct FinalUnitStats {
    bundle: StatsBundle,
    output: Vec<FunctionStats>,
}

impl<'a> From<UnitStats<'a>> for FinalUnitStats {
    fn from(v: UnitStats<'a>) -> Self {
        FinalUnitStats {
            bundle: v.bundle,
            output: v.output,
        }
    }
}

impl Stats {
    fn new_unit_stats(&self) -> UnitStats {
        UnitStats {
            bundle: StatsBundle::default(),
            opt: &self.opt,
            noninline_function_stack: Vec::new(),
            output: Vec::new(),
        }
    }
    fn accumulate(&mut self, mut stats: FinalUnitStats) {
        self.bundle += stats.bundle;
        self.output.append(&mut stats.output);
    }
}

impl<'a> UnitStats<'a> {
    fn enter_noninline_function(
        &mut self,
        name: &MaybeDemangle<'a>,
        unit_dir: &str,
        unit_name: &str,
    ) {
        let demangled = name.demangled();
        self.noninline_function_stack.push(
            if self
                .opt
                .select_functions
                .as_ref()
                .map(|r| r.is_match(&demangled))
                .unwrap_or(true)
            {
                Some(FunctionStats {
                    name: demangled.into_owned(),
                    unit_dir: unit_dir.into(),
                    unit_name: unit_name.into(),
                    stats: VariableStats::default(),
                    variables: Vec::new(),
                })
            } else {
                None
            },
        );
    }
    fn process_variables(&self) -> bool {
        self.noninline_function_stack
            .last()
            .map(|o| o.is_some())
            .unwrap_or(false)
    }
    fn accumulate(
        &mut self,
        var_type: VarType,
        subprogram_name_stack: &[(MaybeDemangle, isize, bool)],
        var_name: Option<MaybeDemangle>,
        var_decl_dir: &str,
        var_decl_file: &str,
        var_decl_line: &str,
        extra_var_info: ExtraVarInfo,
        stats: VariableStats,
    ) {
        if (self.opt.only_parameters && var_type != VarType::Parameter)
            || (self.opt.only_locals && var_type != VarType::Variable)
        {
            return;
        }
        let function_stats = self
            .noninline_function_stack
            .last_mut()
            .unwrap()
            .as_mut()
            .unwrap();
        let mut i = subprogram_name_stack.len();
        while i > 0 && subprogram_name_stack[i - 1].2 {
            i -= 1;
        }
        function_stats.stats += stats.clone();
        match var_type {
            VarType::Parameter => self.bundle.parameters += stats.clone(),
            VarType::Variable => self.bundle.variables += stats.clone(),
        }
        if self.opt.variables {
            function_stats.variables.push(NamedVarStats {
                inlines: subprogram_name_stack[i..]
                    .iter()
                    .map(|&(ref name, _, _)| name.demangled().into_owned())
                    .collect(),
                name: var_name
                    .map(|d| d.demangled())
                    .unwrap_or(Cow::Borrowed("<anon>"))
                    .into_owned(),
                decl_dir: var_decl_dir.into(),
                decl_file: var_decl_file.into(),
                decl_line: var_decl_line.into(),
                extra: extra_var_info,
                stats,
            });
        }
    }
    fn leave_noninline_function(&mut self) {
        if let Some(function_stats) = self.noninline_function_stack.pop().unwrap() {
            if function_stats.stats.instruction_bytes_in_scope > 0
                && (self.opt.functions || self.opt.variables)
            {
                self.output.push(function_stats);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VarType {
    Parameter,
    Variable,
}

#[derive(Clone, Debug, Add, AddAssign, Default)]
pub struct VariableStats {
    pub instruction_bytes_in_scope: u64,
    pub instruction_bytes_covered: u64,
    pub source_lines_in_scope: u64,
    pub source_lines_covered: u64,
}

impl VariableStats {
    pub fn fraction_bytes_covered(&self) -> f64 {
        (self.instruction_bytes_covered as f64) / (self.instruction_bytes_in_scope as f64)
    }
}

fn ranges_instruction_bytes(r: &[gimli::Range]) -> u64 {
    r.iter().fold(0, |sum, r| sum + (r.end - r.begin))
}

fn ranges_end(r: &[gimli::Range]) -> Option<u64> {
    r.iter().last().map(|r| r.end)
}

fn ranges_from_bounds(begin: Option<u64>, end: Option<u64>) -> Vec<gimli::Range> {
    let mut result = Vec::new();
    if let (Some(begin), Some(end)) = (begin, end) {
        if begin < end {
            result.push(gimli::Range { begin, end })
        }
    }
    result
}

fn ranges_overlap(rs1: &[gimli::Range], rs2: &[gimli::Range]) -> Vec<gimli::Range> {
    let mut iter1 = rs1.iter();
    let mut iter2 = rs2.iter();
    let mut r1_opt: Option<gimli::Range> = iter1.next().map(|r| *r);
    let mut r2_opt: Option<gimli::Range> = iter2.next().map(|r| *r);
    let mut result = Vec::new();
    while let (Some(r1), Some(r2)) = (r1_opt, r2_opt) {
        let overlap_start = max(r1.begin, r2.begin);
        let overlap_end = min(r1.end, r2.end);
        if overlap_start < overlap_end {
            result.push(gimli::Range {
                begin: overlap_start,
                end: overlap_end,
            });
        }
        let new_min = overlap_end;
        r1_opt = if r1.end <= new_min {
            iter1.next().map(|r| *r)
        } else {
            Some(gimli::Range {
                begin: max(r1.begin, new_min),
                end: r1.end,
            })
        };
        r2_opt = if r2.end <= new_min {
            iter2.next().map(|r| *r)
        } else {
            Some(gimli::Range {
                begin: max(r2.begin, new_min),
                end: r2.end,
            })
        };
    }
    result
}

fn ranges_source_lines<R: Reader>(
    ranges: &[gimli::Range],
    line_program: gimli::IncompleteLineNumberProgram<R>,
) -> (u64, BTreeSet<u64>) {
    let mut line_sm = line_program.rows();
    let mut row = line_sm
        .next_row()
        .unwrap()
        .expect("Next row should exist in line table")
        .1;
    let mut previous_row_line = None;
    // Instructions may mark out non-contiguous, overlapping source line ranges.
    // A source line set allows accurate counting even with overlaps.
    let mut source_line_set = BTreeSet::new();
    for range in ranges {
        // println!("Range: [{:#x}, {:#x})", range.begin, range.end);
        loop {
            // Continue until we find a row for beginning of range (may not be exact match)
            if row.address() < range.begin {
                previous_row_line = row.line();
                row = line_sm
                    .next_row()
                    .unwrap()
                    .expect("Next row should exist in line table")
                    .1;
                continue;
            }
            // If the start of the instruction range is between two line table rows,
            // include the line from the previous row as well.
            if let Some(line) = previous_row_line {
                if row.address() > range.begin {
                    // println!("Line: {:#x} -> {:?}", range.begin, Some(line));
                    source_line_set.insert(line);
                }
                previous_row_line = None;
            }
            // The end of an instruction range is exclusive, stop when reached
            if row.address() >= range.end {
                break;
            }
            // println!("Line: {:#x} -> {:?}", row.address(), row.line());
            if let Some(line) = row.line() {
                source_line_set.insert(line);
            }
            row = line_sm
                .next_row()
                .unwrap()
                .expect("Next row should exist in line table")
                .1;
        }
    }
    // println!("Total: {}", source_line_set.len());
    (source_line_set.len() as u64, source_line_set)
}

fn sort_nonoverlapping(rs: &mut [gimli::Range]) {
    rs.sort_by_key(|r| r.begin);
    for r in 1..rs.len() {
        assert!(rs[r - 1].end <= rs[r].begin);
    }
}

fn to_ref_str<'abbrev, 'unit, R>(
    unit: &'unit CompilationUnitHeader<R, R::Offset>,
    entry: &gimli::DebuggingInformationEntry<'abbrev, 'unit, R>,
) -> String
where
    R: Reader,
{
    format!("{:x}:{:x}", unit.offset().0, entry.offset().0)
}

enum MaybeDemangle<'a> {
    Demangle(Cow<'a, str>),
    Raw(Cow<'a, str>),
}

impl<'a> MaybeDemangle<'a> {
    fn demangled(&self) -> Cow<'a, str> {
        match self {
            &MaybeDemangle::Demangle(ref s) => {
                if let Ok(sym) = BorrowedSymbol::new(s.as_bytes()) {
                    match sym.demangle(&DemangleOptions::default()) {
                        Ok(d) => d.into(),
                        Err(_) => s.clone(),
                    }
                } else {
                    s.clone()
                }
            }
            &MaybeDemangle::Raw(ref s) => s.clone(),
        }
    }
}

fn lookup_multiple<'abbrev, 'unit, 'a>(
    unit: &'unit CompilationUnitHeader<EndianSlice<'a, gimli::LittleEndian>, usize>,
    entry: &gimli::DebuggingInformationEntry<'abbrev, 'unit, EndianSlice<'a, gimli::LittleEndian>>,
    abbrevs: &gimli::Abbreviations,
    attrs: &[gimli::DwAt],
) -> Option<(
    gimli::DwAt,
    AttributeValue<EndianSlice<'a, gimli::LittleEndian>>,
)>
where
    'a: 'unit,
{
    let mut entry = entry.clone();
    loop {
        for attr in attrs {
            let value = entry.attr_value(*attr).unwrap();
            if let Some(value) = value {
                return Some((*attr, value));
            }
        }
        let reference = if let Some(r) = entry.attr_value(gimli::DW_AT_abstract_origin).unwrap() {
            r
        } else if let Some(r) = entry.attr_value(gimli::DW_AT_specification).unwrap() {
            r
        } else {
            return None;
        };
        match reference {
            gimli::AttributeValue::UnitRef(offset) => {
                entry = unit
                    .entries_at_offset(abbrevs, offset)
                    .unwrap()
                    .next_dfs()
                    .unwrap()
                    .unwrap()
                    .1
                    .clone();
            }
            _ => {
                panic!("Unexpected attribute value for reference: {:?}", reference);
            }
        }
    }
}

fn lookup<'abbrev, 'unit, 'a>(
    unit: &'unit CompilationUnitHeader<EndianSlice<'a, gimli::LittleEndian>, usize>,
    entry: &gimli::DebuggingInformationEntry<'abbrev, 'unit, EndianSlice<'a, gimli::LittleEndian>>,
    abbrevs: &gimli::Abbreviations,
    attr: gimli::DwAt,
) -> Option<AttributeValue<EndianSlice<'a, gimli::LittleEndian>>>
where
    'a: 'unit,
{
    lookup_multiple(unit, entry, abbrevs, &[attr]).map(|av| av.1)
}

fn lookup_name<'abbrev, 'unit, 'a>(
    unit: &'unit CompilationUnitHeader<EndianSlice<'a, gimli::LittleEndian>, usize>,
    entry: &gimli::DebuggingInformationEntry<'abbrev, 'unit, EndianSlice<'a, gimli::LittleEndian>>,
    abbrevs: &gimli::Abbreviations,
    debug_str: &'a gimli::DebugStr<EndianSlice<'a, gimli::LittleEndian>>,
) -> Option<MaybeDemangle<'a>>
where
    'a: 'unit,
{
    let result = lookup_multiple(
        unit,
        entry,
        abbrevs,
        &[gimli::DW_AT_linkage_name, gimli::DW_AT_name],
    );
    match result {
        Some((gimli::DW_AT_linkage_name, gimli::AttributeValue::String(string))) => {
            Some(MaybeDemangle::Demangle(string.to_string_lossy()))
        }
        Some((gimli::DW_AT_linkage_name, gimli::AttributeValue::DebugStrRef(offset))) => Some(
            MaybeDemangle::Demangle(debug_str.get_str(offset).unwrap().to_string_lossy()),
        ),
        Some((gimli::DW_AT_name, gimli::AttributeValue::String(string))) => {
            Some(MaybeDemangle::Raw(string.to_string_lossy()))
        }
        Some((gimli::DW_AT_name, gimli::AttributeValue::DebugStrRef(offset))) => Some(
            MaybeDemangle::Raw(debug_str.get_str(offset).unwrap().to_string_lossy()),
        ),
        Some(_) => panic!("Invalid DW_AT_name"),
        None => None,
    }
}

fn is_allowed_expression<'a>(
    mut e: gimli::Evaluation<EndianSlice<'a, gimli::LittleEndian>>,
    no_entry_value: bool,
    no_parameter_ref: bool,
) -> bool {
    match e.evaluate() {
        Ok(gimli::EvaluationResult::RequiresEntryValue(_)) => !no_entry_value,
        Ok(gimli::EvaluationResult::RequiresParameterRef(_)) => !no_parameter_ref,
        _ => true,
    }
}

pub fn evaluate_info<'a>(
    debug_info: &'a gimli::DebugInfo<EndianSlice<'a, gimli::LittleEndian>>,
    debug_abbrev: &'a gimli::DebugAbbrev<EndianSlice<'a, gimli::LittleEndian>>,
    debug_str: &'a gimli::DebugStr<EndianSlice<'a, gimli::LittleEndian>>,
    rnglists: &gimli::RangeLists<EndianSlice<'a, gimli::LittleEndian>>,
    loclists: &gimli::LocationLists<EndianSlice<'a, gimli::LittleEndian>>,
    debug_line: &'a gimli::DebugLine<EndianSlice<'a, gimli::LittleEndian>>,
    no_entry_value: bool,
    no_parameter_ref: bool,
    stats: &'a mut Stats,
) {
    let units = debug_info.units().collect::<Vec<_>>().unwrap();
    let process_unit = |stats: &Stats,
                        unit: CompilationUnitHeader<
        EndianSlice<'a, gimli::LittleEndian>,
        usize,
    >|
     -> FinalUnitStats {
        let mut unit_stats = stats.new_unit_stats();
        let abbrevs = unit.abbreviations(debug_abbrev).unwrap();
        let mut entries = unit.entries(&abbrevs);
        let mut base_address = None;
        let mut line_program_offset = None;
        let mut unit_dir = None;
        let mut unit_name = None;
        {
            let (delta, entry) = entries.next_dfs().unwrap().unwrap();
            assert_eq!(delta, 0);
            if let Some(gimli::AttributeValue::Addr(addr)) =
                entry.attr_value(gimli::DW_AT_low_pc).unwrap()
            {
                base_address = Some(addr);
            }
            let producer = match entry.attr_value(gimli::DW_AT_producer).unwrap() {
                Some(gimli::AttributeValue::String(string)) => string.to_string_lossy(),
                Some(gimli::AttributeValue::DebugStrRef(offset)) => {
                    debug_str.get_str(offset).unwrap().to_string_lossy()
                }
                Some(_) => panic!("Invalid DW_AT_producer"),
                None => Cow::Borrowed(""),
            };
            let language = if producer.contains("rustc version") {
                Language::Rust
            } else {
                Language::Cpp
            };
            if stats.opt.language.map(|l| l != language).unwrap_or(false) {
                return unit_stats.into();
            }
            if let Some(gimli::AttributeValue::DebugLineRef(offset)) =
                entry.attr_value(gimli::DW_AT_stmt_list).unwrap()
            {
                line_program_offset = Some(offset);
            }
            if let Some(dir) = entry.attr(gimli::DW_AT_comp_dir).unwrap() {
                unit_dir = dir.string_value(debug_str);
            }
            if let Some(name) = entry.attr(gimli::DW_AT_name).unwrap() {
                unit_name = name.string_value(debug_str);
            }
        }
        let unit_dir_for_output = unit_dir.map_or(Cow::Borrowed("<unknown directory>"), |n| {
            n.to_string_lossy()
        });
        let unit_name_for_output =
            unit_name.map_or(Cow::Borrowed("<unknown unit>"), |n| n.to_string_lossy());
        let line_program = line_program_offset
            .map(|offset| {
                debug_line
                    .program(offset, unit.address_size(), unit_dir, unit_name)
                    .unwrap()
            })
            .expect("Debug info should have source line table");
        let mut depth = 0;
        let mut scopes: Vec<(Vec<gimli::Range>, isize)> = Vec::new();
        let mut namespace_stack: Vec<(MaybeDemangle, isize, bool)> = Vec::new();
        loop {
            let (delta, entry) = match entries.next_dfs().unwrap() {
                None => break,
                Some(entry) => entry,
            };
            depth += delta;
            while scopes.last().map(|v| v.1 >= depth).unwrap_or(false) {
                scopes.pop();
            }
            while namespace_stack
                .last()
                .map(|v| v.1 >= depth)
                .unwrap_or(false)
            {
                if !namespace_stack.pop().unwrap().2 {
                    unit_stats.leave_noninline_function();
                }
            }
            if let Some(AttributeValue::RangeListsRef(offset)) =
                entry.attr_value(gimli::DW_AT_ranges).unwrap()
            {
                let rs = rnglists
                    .ranges(
                        offset,
                        unit.version(),
                        unit.address_size(),
                        base_address.unwrap(),
                    )
                    .unwrap();
                let mut bytes_ranges = rs.collect::<Vec<_>>().unwrap();
                sort_nonoverlapping(&mut bytes_ranges);
                scopes.push((bytes_ranges, depth));
            } else if let Some(AttributeValue::Udata(data)) =
                entry.attr_value(gimli::DW_AT_high_pc).unwrap()
            {
                if let Some(gimli::AttributeValue::Addr(addr)) =
                    entry.attr_value(gimli::DW_AT_low_pc).unwrap()
                {
                    let bytes_range = gimli::Range {
                        begin: addr,
                        end: addr + data,
                    };
                    scopes.push((vec![bytes_range], depth));
                }
            }
            let var_type = match entry.tag() {
                gimli::DW_TAG_formal_parameter => VarType::Parameter,
                gimli::DW_TAG_variable => VarType::Variable,
                gimli::DW_TAG_subprogram => {
                    if let Some(name) = lookup_name(&unit, &entry, &abbrevs, debug_str) {
                        unit_stats.enter_noninline_function(
                            &name,
                            &unit_dir_for_output,
                            &unit_name_for_output,
                        );
                        namespace_stack.push((name, depth, false));
                    }
                    continue;
                }
                gimli::DW_TAG_inlined_subroutine => {
                    if let Some(name) = lookup_name(&unit, &entry, &abbrevs, debug_str) {
                        namespace_stack.push((name, depth, true));
                    }
                    continue;
                }
                _ => continue,
            };
            if !unit_stats.process_variables() {
                continue;
            }
            let ranges = if let Some(s) = scopes.last() {
                if s.1 + 1 == depth && !s.0.is_empty() {
                    &s.0[..]
                } else {
                    continue;
                }
            } else {
                continue;
            };
            let var_name = lookup_name(&unit, &entry, &abbrevs, debug_str);
            let var_decl_dir = match lookup(&unit, &entry, &abbrevs, gimli::DW_AT_decl_file) {
                Some(gimli::AttributeValue::FileIndex(file)) => {
                    line_program
                        .header()
                        .file(file)
                        .unwrap()
                        .directory(line_program.header())
                        .unwrap()
                        .to_string_lossy()
                }
                Some(_) => panic!("Invalid DW_AT_decl_file"),
                None => Cow::Borrowed("<unknown directory>"),
            };
            let var_decl_file = match lookup(&unit, &entry, &abbrevs, gimli::DW_AT_decl_file) {
                Some(gimli::AttributeValue::FileIndex(file)) => line_program
                    .header()
                    .file(file)
                    .unwrap()
                    .path_name()
                    .to_string_lossy(),
                Some(_) => panic!("Invalid DW_AT_decl_file"),
                None => Cow::Borrowed("<unknown file>"),
            };
            let var_decl_line = match lookup(&unit, &entry, &abbrevs, gimli::DW_AT_decl_line) {
                Some(gimli::AttributeValue::Udata(line)) => line.to_string(),
                Some(_) => panic!("Invalid DW_AT_decl_line"),
                None => String::from("<unknown line>"),
            };
            // println!("Variable stats for {}", var_name.as_ref().unwrap().demangled());
            let (var_stats, extra_var_info) = if entry
                .attr_value(gimli::DW_AT_const_value)
                .unwrap()
                .is_some()
            {
                let bytes_in_scope = ranges_instruction_bytes(ranges);
                let (lines_in_scope, source_line_set) =
                    ranges_source_lines(ranges, line_program.clone());
                (
                    VariableStats {
                        instruction_bytes_in_scope: bytes_in_scope,
                        instruction_bytes_covered: bytes_in_scope,
                        source_lines_in_scope: lines_in_scope,
                        source_lines_covered: lines_in_scope,
                    },
                    ExtraVarInfo {
                        source_line_set_covered: source_line_set,
                        source_line_set_after_covered: None,
                    },
                )
            } else {
                match entry.attr_value(gimli::DW_AT_location).unwrap() {
                    Some(AttributeValue::Exprloc(expr)) => {
                        let bytes_in_scope = ranges_instruction_bytes(ranges);
                        let (lines_in_scope, source_line_set) =
                            ranges_source_lines(ranges, line_program.clone());
                        let (bytes_covered, lines_covered, source_line_set) = if (no_entry_value
                            || no_parameter_ref)
                            && !is_allowed_expression(
                                expr.evaluation(unit.address_size(), unit.format()),
                                no_entry_value,
                                no_parameter_ref,
                            ) {
                            (0, 0, BTreeSet::new())
                        } else {
                            (bytes_in_scope, lines_in_scope, source_line_set)
                        };
                        (
                            VariableStats {
                                instruction_bytes_in_scope: bytes_in_scope,
                                instruction_bytes_covered: bytes_covered,
                                source_lines_in_scope: lines_in_scope,
                                source_lines_covered: lines_covered,
                            },
                            ExtraVarInfo {
                                source_line_set_covered: source_line_set,
                                source_line_set_after_covered: None,
                            },
                        )
                    }
                    Some(AttributeValue::LocationListsRef(loc)) => {
                        let mut locations = {
                            let iter = loclists
                                .locations(
                                    loc,
                                    unit.version(),
                                    unit.address_size(),
                                    base_address.unwrap(),
                                )
                                .expect("invalid location list");
                            iter.filter_map(|e| {
                                if (no_entry_value || no_parameter_ref)
                                    && !is_allowed_expression(
                                        e.data.evaluation(unit.address_size(), unit.format()),
                                        no_entry_value,
                                        no_parameter_ref,
                                    )
                                {
                                    None
                                } else {
                                    Some(e.range)
                                }
                            })
                            .collect::<Vec<_>>()
                            .expect("invalid location list")
                        };
                        sort_nonoverlapping(&mut locations);
                        let covered_ranges = ranges_overlap(ranges, &locations[..]);
                        let (source_lines_covered, source_line_set_covered) =
                            ranges_source_lines(&covered_ranges, line_program.clone());
                        let after_covered_ranges =
                            ranges_from_bounds(ranges_end(&covered_ranges), ranges_end(ranges));
                        let (_, source_line_set_after_covered) =
                            ranges_source_lines(&after_covered_ranges, line_program.clone());
                        (
                            VariableStats {
                                instruction_bytes_in_scope: ranges_instruction_bytes(ranges),
                                instruction_bytes_covered: ranges_instruction_bytes(
                                    &covered_ranges,
                                ),
                                source_lines_in_scope: ranges_source_lines(
                                    ranges,
                                    line_program.clone(),
                                )
                                .0,
                                source_lines_covered,
                            },
                            ExtraVarInfo {
                                source_line_set_covered,
                                source_line_set_after_covered: Some(source_line_set_after_covered),
                            },
                        )
                    }
                    None => (
                        VariableStats {
                            instruction_bytes_in_scope: ranges_instruction_bytes(ranges),
                            instruction_bytes_covered: 0,
                            source_lines_in_scope: ranges_source_lines(
                                ranges,
                                line_program.clone(),
                            )
                            .0,
                            source_lines_covered: 0,
                        },
                        ExtraVarInfo {
                            source_line_set_covered: BTreeSet::new(),
                            source_line_set_after_covered: None,
                        },
                    ),
                    _ => panic!(
                        "Unknown DW_AT_location attribute at {}",
                        to_ref_str(&unit, &entry)
                    ),
                }
            };
            unit_stats.accumulate(
                var_type,
                &namespace_stack,
                var_name,
                &var_decl_dir,
                &var_decl_file,
                &var_decl_line,
                extra_var_info,
                var_stats,
            );
        }
        unit_stats.into()
    };
    let all_stats = units
        .into_par_iter()
        .map(|u| process_unit(stats, u))
        .collect::<Vec<_>>();
    for s in all_stats {
        stats.accumulate(s);
    }
}
