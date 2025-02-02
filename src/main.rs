use std::borrow::{Borrow, Cow};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::iter::Iterator;
use std::path::Path;
use std::process;

use debuginfo_quality::*;
use object::{Object, ObjectSection};
use path_absolutize::Absolutize;
use structopt::StructOpt;
use typed_arena::Arena;

fn map(path: &Path) -> memmap2::Mmap {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to open file '{}': {}", path.display(), &err);
            process::exit(1);
        }
    };
    match unsafe { memmap2::Mmap::map(&file) } {
        Ok(mmap) => mmap,
        Err(err) => {
            eprintln!("Failed to map file '{}': {}", path.display(), &err);
            process::exit(1);
        }
    }
}

fn open<'a>(path: &Path, mmap: &'a memmap2::Mmap) -> object::File<'a> {
    let file = match object::File::parse(&**mmap) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to parse file '{}': {}", path.display(), err);
            process::exit(1);
        }
    };
    assert!(file.is_little_endian());
    file
}

fn write_stats<W: io::Write>(
    mut w: W,
    stats: &VariableStats,
    base_stats: Option<&VariableStats>,
    adj_lines: Option<u64>,
    flt_lines: Option<u64>,
    src_scope_lines: Option<u64>,
) {
    assert!(base_stats.is_none() || adj_lines.is_none());
    if let Some(b) = base_stats {
        writeln!(
            w,
            "\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}",
            stats.instruction_bytes_covered,
            stats.instruction_bytes_in_scope,
            b.instruction_bytes_covered,
            b.instruction_bytes_in_scope,
            stats.source_lines_covered,
            stats.source_lines_in_scope,
            b.source_lines_covered,
            b.source_lines_in_scope
        )
        .unwrap();
    } else {
        write!(
            w,
            "\t{:12}\t{:12}\t{:12}\t{:12}",
            stats.instruction_bytes_covered,
            stats.instruction_bytes_in_scope,
            stats.source_lines_covered,
            stats.source_lines_in_scope
        )
        .unwrap();
        if let Some(adj_lines) = adj_lines {
            write!(w, "\t{:12}", adj_lines).unwrap();
        }
        if let Some(flt_lines) = flt_lines {
            write!(w, "\t{:12}", flt_lines).unwrap();
        }
        if let Some(src_scope_lines) = src_scope_lines {
            write!(w, "\t{:12}", src_scope_lines).unwrap();
        }
        writeln!(w).unwrap();
    }
}

fn write_stats_label<W: io::Write>(
    mut w: W,
    label: &str,
    stats: &VariableStats,
    base_stats: Option<&VariableStats>,
    opt: &Opt,
) {
    write!(w, "{}", label).unwrap();
    if !opt.tsv && (opt.functions || opt.variables) {
        write!(&mut w, "\t\t\t\t").unwrap();
    }
    write_stats(w, stats, base_stats, None, None, None);
}

// TODO: Use references instead of copying
#[derive(Debug)]
struct RegionLocation {
    file: String,
    line: u64,
}

impl RegionLocation {
    fn parse(location: &str) -> Result<Self, Box<dyn Error>> {
        let mut parts = location.split(':');
        let file = parts.next().unwrap().to_string();
        let line = parts.next().unwrap().parse()?;
        Ok(Self { file, line })
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RegionKind {
    Unsupported,
    Computation,
    DeclScope,
    MayBeDefined,
    MustBeDefined,
}

#[derive(Debug)]
struct Region {
    start: RegionLocation,
    end: RegionLocation,
    kind: RegionKind,
    detail: String,
}

impl Region {
    fn parse(line: &str) -> Result<Self, Box<dyn Error>> {
        // Line format:
        // start as `file:line:column`\t
        // end as `file:line:column`\t
        // kind (`Computation`, `MustBeDefined`)\t
        // detail tail (e.g. expression type, variable description)
        let mut parts = line.split('\t');
        let start = RegionLocation::parse(parts.next().unwrap())?;
        let end = RegionLocation::parse(parts.next().unwrap())?;
        let kind = match parts.next().unwrap() {
            "Computation" => RegionKind::Computation,
            "DeclScope" => RegionKind::DeclScope,
            "MayBeDefined" => RegionKind::MayBeDefined,
            "MustBeDefined" => RegionKind::MustBeDefined,
            _ => RegionKind::Unsupported,
        };
        let detail = parts.next().unwrap().to_owned();
        Ok(Self {
            start,
            end,
            kind,
            detail,
        })
    }
}

fn parse_regions(bytes: &[u8]) -> Result<Vec<Region>, Box<dyn Error>> {
    let mut regions = Vec::new();

    for regions_line in bytes.lines() {
        let regions_line = regions_line.unwrap();
        let region = Region::parse(&regions_line)?;
        regions.push(region);
    }

    Ok(regions)
}

fn computation_line_sets_by_file(regions: &Vec<Region>) -> HashMap<String, BTreeSet<u64>> {
    let mut line_sets_by_file = HashMap::new();

    for region in regions {
        if region.kind != RegionKind::Computation {
            continue;
        }

        let file = region.start.file.clone();
        // Normalise on insert and access to allow for relative paths
        let canonical_file = Path::new(&file)
            .absolutize()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let line_set: &mut BTreeSet<u64> = line_sets_by_file.entry(canonical_file).or_default();
        for line in region.start.line..=region.end.line {
            line_set.insert(line);
        }
    }

    line_sets_by_file
}

fn first_defined_line_by_variable(regions: &Vec<Region>) -> HashMap<String, u64> {
    let mut line_by_variable = HashMap::new();

    for region in regions {
        if region.kind != RegionKind::MayBeDefined && region.kind != RegionKind::MustBeDefined {
            continue;
        }

        let variable = region.detail.clone();
        let stored_line: &mut u64 = line_by_variable.entry(variable).or_insert(u64::MAX);
        let current_line = &*stored_line;
        let min_line = current_line.min(&region.start.line);
        *stored_line = *min_line;
    }

    line_by_variable
}

fn scope_line_sets_by_variable(regions: &Vec<Region>) -> HashMap<String, BTreeSet<u64>> {
    let mut line_sets_by_variable = HashMap::new();

    for region in regions {
        if region.kind != RegionKind::DeclScope {
            continue;
        }

        let variable = region.detail.clone();
        let line_set: &mut BTreeSet<u64> = line_sets_by_variable.entry(variable).or_default();
        for line in region.start.line..=region.end.line {
            line_set.insert(line);
        }
    }

    line_sets_by_variable
}

fn variables_by_function(regions: &Vec<Region>) -> HashMap<(String, String), HashSet<String>> {
    let mut variables_by_function = HashMap::new();

    for region in regions {
        if region.kind != RegionKind::DeclScope {
            continue;
        }

        // TODO: Likely needs special handling for languages like C++ that could have multiple of
        // same function name in the same source file
        let variable = region.detail.clone();
        let mut parts = variable.split(", ");
        let leaf_function_name = parts.next().unwrap().to_owned();
        let decl = parts.nth(1).unwrap();
        let decl_file = decl.split(':').next().unwrap().to_owned();

        let variable_set: &mut HashSet<String> = variables_by_function
            .entry((leaf_function_name, decl_file))
            .or_default();
        variable_set.insert(variable);
    }

    variables_by_function
}

fn cachegrind_line_sets_by_file(bytes: &[u8]) -> HashMap<String, BTreeSet<u64>> {
    let mut line_sets_by_file = HashMap::new();

    let mut file = None;

    for cg_line in bytes.lines() {
        let cg_line = cg_line.unwrap();

        let mut parts = cg_line.split('=');
        if parts.next() == Some("fl") {
            file = parts.next().map(|s| s.to_owned());
            // println!("File: {:?}", &file);
            continue;
        }

        parts = cg_line.split(' ');
        let line = parts.next().unwrap().parse::<u64>();
        if let Ok(line) = line {
            if line == 0 {
                continue;
            }
            let file = file.clone().unwrap();
            let line_set: &mut BTreeSet<u64> = line_sets_by_file.entry(file).or_default();
            line_set.insert(line);
            // println!("Line: {}", &line);
            continue;
        }
    }

    line_sets_by_file
}

fn scope_line_set_for_variable(
    opt: &Opt,
    scope_line_sets_by_variable: &Option<HashMap<String, BTreeSet<u64>>>,
    variable_description: &String,
    computation_line_set: Option<&BTreeSet<u64>>,
    first_defined_line: Option<&u64>,
) -> Option<BTreeSet<u64>> {
    let scope_line_sets_by_variable = scope_line_sets_by_variable.as_ref().unwrap();
    let mut scope_line_set = scope_line_sets_by_variable
        .get(variable_description)
        .cloned();
    if opt.only_computation_regions {
        if let Some(computation_line_set) = computation_line_set {
            scope_line_set =
                scope_line_set.map(|set| set.intersection(computation_line_set).cloned().collect());
        } else {
            scope_line_set.as_mut().map(|set| {
                set.clear();
            });
        }
    }
    if opt.range_start_first_defined_region {
        if let Some(first_defined_line) = first_defined_line {
            scope_line_set.as_mut().map(|set| {
                while set.first().unwrap_or(&u64::MAX) < first_defined_line {
                    set.pop_first();
                }
            });
        } else {
            scope_line_set.as_mut().map(|set| {
                set.clear();
            });
        }
    }
    scope_line_set
}

fn goodness(&(ref a, ref a_base): &(FunctionStats, Option<&FunctionStats>)) -> (f64, i64) {
    (
        if let Some(a_base) = a_base.as_ref() {
            a.stats.fraction_bytes_covered() - a_base.stats.fraction_bytes_covered()
        } else {
            a.stats.fraction_bytes_covered()
        },
        -(a.stats.instruction_bytes_in_scope as i64),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    if opt.baseline.is_none() && (opt.no_entry_value_baseline || opt.no_parameter_ref_baseline) {
        eprintln!("Don't specify baseline options with no baseline!");
        process::exit(1);
    }

    let file_map = map(&opt.file);
    let file = open(&opt.file, &file_map);
    let baseline_map = opt.baseline.as_ref().map(|p| (p, map(p)));
    let baseline_file = baseline_map.as_ref().map(|&(ref p, ref m)| open(p, m));

    fn load_section<'a, 'file, 'input, S>(
        arena: &'a Arena<Cow<'file, [u8]>>,
        file: &'file object::File<'input>,
    ) -> S
    where
        S: gimli::Section<gimli::EndianSlice<'a, gimli::LittleEndian>>,
        'file: 'input,
        'a: 'file,
    {
        let data = match file.section_by_name(S::section_name()) {
            Some(ref section) => section
                .uncompressed_data()
                .unwrap_or(Cow::Borrowed(&[][..])),
            None => Cow::Borrowed(&[][..]),
        };
        let data_ref = (*arena.alloc(data)).borrow();
        S::from(gimli::EndianSlice::new(data_ref, gimli::LittleEndian))
    }

    let mut stats = Stats::new(opt.clone());
    let mut base_stats = None;

    {
        let file = &file;
        let arena = Arena::new();

        // Variables representing sections of the file. The type of each is inferred from its use in the
        // validate_info function below.
        let debug_abbrev = &load_section(&arena, file);
        let debug_info = &load_section(&arena, file);
        let debug_ranges = load_section(&arena, file);
        let debug_rnglists = load_section(&arena, file);
        let rnglists = &gimli::RangeLists::new(debug_ranges, debug_rnglists).unwrap();
        let debug_str = &load_section(&arena, file);
        let debug_loc = load_section(&arena, file);
        let debug_loclists = load_section(&arena, file);
        let loclists = &gimli::LocationLists::new(debug_loc, debug_loclists).unwrap();
        let debug_line = &load_section(&arena, file);

        evaluate_info(
            debug_info,
            debug_abbrev,
            debug_str,
            rnglists,
            loclists,
            debug_line,
            stats.opt.no_entry_value,
            stats.opt.no_parameter_ref,
            &mut stats,
        );
    }

    if let Some(file) = baseline_file.as_ref() {
        let arena = Arena::new();

        // Variables representing sections of the file. The type of each is inferred from its use in the
        // validate_info function below.
        let debug_abbrev = &load_section(&arena, file);
        let debug_info = &load_section(&arena, file);
        let debug_ranges = load_section(&arena, file);
        let debug_rnglists = load_section(&arena, file);
        let rnglists = &gimli::RangeLists::new(debug_ranges, debug_rnglists)?;
        let debug_str = &load_section(&arena, file);
        let debug_loc = load_section(&arena, file);
        let debug_loclists = load_section(&arena, file);
        let loclists = &gimli::LocationLists::new(debug_loc, debug_loclists)?;
        let debug_line = &load_section(&arena, file);

        let mut stats = Stats::new(opt.clone());
        evaluate_info(
            debug_info,
            debug_abbrev,
            debug_str,
            rnglists,
            loclists,
            debug_line,
            stats.opt.no_entry_value_baseline,
            stats.opt.no_parameter_ref_baseline,
            &mut stats,
        );
        base_stats = Some(stats);
    }

    let regions_map = opt.regions.as_ref().map(|path| map(path));
    let regions = regions_map.map(|r| parse_regions(&r));
    let computation_line_sets_by_file = regions
        .as_ref()
        .map(|r| computation_line_sets_by_file(r.as_ref().unwrap()));
    let first_defined_line_by_variable = regions
        .as_ref()
        .map(|r| first_defined_line_by_variable(r.as_ref().unwrap()));
    let scope_line_sets_by_variable = regions
        .as_ref()
        .map(|r| scope_line_sets_by_variable(r.as_ref().unwrap()));
    let variables_by_function = regions
        .as_ref()
        .map(|r| variables_by_function(r.as_ref().unwrap()));

    let cachegrind_map = opt.cachegrind.as_ref().map(|path| map(path));
    let cachegrind_line_sets_by_file = cachegrind_map.map(|c| cachegrind_line_sets_by_file(&c));

    let adjusting_by_baseline = stats.opt.range_start_baseline || stats.opt.extend_from_baseline;
    let filtering_by_regions = stats.opt.only_computation_regions
        || stats.opt.range_start_first_defined_region
        || stats.opt.only_cachegrind_regions;

    // Lines mode currently assumes debug info only knows about a single file
    let mut lines_src_file = None;
    let mut locatable_vars_by_line: Option<BTreeMap<u64, HashSet<String>>> = None;
    let mut scope_vars_by_line: Option<BTreeMap<u64, HashSet<String>>> = None;
    if stats.opt.lines {
        locatable_vars_by_line = Some(BTreeMap::new());
        scope_vars_by_line = Some(BTreeMap::new());
    }

    let stdout = io::stdout();
    let mut stdout_locked = stdout.lock();
    let mut w = BufWriter::new(&mut stdout_locked);

    // TODO: In need of reorganisation, perhaps with functions for each report step
    // Too many conditionals for all the possible combinations of options

    if stats.opt.lines {
        // Line mode
        writeln!(
            &mut w,
            "{:12}\t{:12}\t{:12}\t{:12}\t{:12}",
            "Line", "Present", "Reachable", "Locatable (V)", "Scope (V)"
        )?;
    } else {
        // Function / variable modes
        if !stats.opt.tsv && (stats.opt.functions || stats.opt.variables) {
            write!(&mut w, "\t\t\t\t")?;
        }
        write!(&mut w, "Name\tInstance")?;
        if base_stats.is_some() && !adjusting_by_baseline {
            writeln!(
                &mut w,
                "\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}",
                "Cov (B)",
                "Scope (B)",
                "BaseCov (B)",
                "BaseScope (B)",
                "Cov (L)",
                "Scope (L)",
                "BaseCov (L)",
                "BaseScope (L)"
            )?;
        } else {
            write!(
                &mut w,
                "\t{:12}\t{:12}\t{:12}\t{:12}",
                "Cov (B)", "Scope (B)", "Cov (L)", "Scope (L)"
            )?;
            if adjusting_by_baseline {
                write!(&mut w, "\t{:12}", "Adj Cov (L)")?;
            }
            if filtering_by_regions {
                write!(&mut w, "\t{:12}", "Flt Cov (L)")?;
            }
            if stats.opt.scope_regions {
                write!(&mut w, "\t{:12}", "Src Scope (L)")?;
            }
            writeln!(&mut w)?;
        }
    }
    writeln!(&mut w)?;

    if stats.opt.functions || stats.opt.variables || stats.opt.lines {
        let mut functions: Vec<(FunctionStats, Option<&FunctionStats>)> =
            if let Some(base) = base_stats.as_ref() {
                let mut base_functions = HashMap::new();
                // TODO: Avoid cloning keys here...?
                for f in base.output.iter() {
                    base_functions.insert((f.name.clone(), f.unit_name.clone()), f);
                }
                stats
                    .output
                    .into_iter()
                    .filter_map(|o| {
                        base_functions
                            .get(&(o.name.clone(), o.unit_name.clone()))
                            .map(|b| (o, Some(*b)))
                    })
                    .collect()
            } else {
                stats.output.into_iter().map(|o| (o, None)).collect()
            };

        functions.sort_by(|a, b| goodness(a).partial_cmp(&goodness(b)).unwrap());

        // TODO: Support multiple source files in line mode
        let mut cachegrind_line_set = None;

        // TODO: Supplement function list using source analysis when present
        // This would allow reporting status of variables from functions not emitted in DWARF
        for (function_stats, base_function_stats) in functions {
            if stats.opt.variables || stats.opt.lines {
                let unit_name = function_stats.unit_name.clone();

                // Track variables we encountered
                // This allows a later stage to compare this with source analysis and
                // report any that may not have been emitted in DWARF
                let mut encountered_variables_by_function: Option<
                    HashMap<(String, Vec<String>, String), HashSet<String>>,
                > = None;
                if stats.opt.scope_regions {
                    encountered_variables_by_function = Some(HashMap::new());
                }

                for v in function_stats.variables {
                    // Check declaration file in lines mode
                    if stats.opt.lines {
                        if let Some(ref lines_src_file) = lines_src_file {
                            assert!(
                                *lines_src_file == v.decl_file,
                                "Lines mode currently assumes a single source file"
                            );
                        } else {
                            lines_src_file = Some(v.decl_file.clone());
                        }
                    }

                    let base_v = base_function_stats.and_then(|bf| {
                        let mut same_v = bf.variables.iter().filter(|&bv| {
                            bv.name == v.name
                                && bv.decl_file == v.decl_file
                                && bv.decl_line == v.decl_line
                                && bf.unit_name == unit_name
                        });
                        if same_v.clone().count() == 1 {
                            same_v.next()
                        } else {
                            None
                        }
                    });

                    // Separate any inline ancestors from the leaf function that contains the
                    // variable at the source level
                    let (leaf_function_name, mut inline_ancestors) = {
                        let mut names = vec![function_stats.name.clone()];
                        names.append(&mut v.inlines.clone());
                        let leaf = names.pop().unwrap();
                        (leaf, names)
                    };

                    // Matching with source analysis uses the function containing the variable
                    // Without inlining, there's only one function, so the answer is clear.
                    // With inlining, we want the leaf function name at the end of the inlines.
                    let mut variable_description = leaf_function_name.clone();
                    // Source analysis can't produce a consistent unit name
                    // Rely on absolute file paths below to distinguish similarly named files
                    // in different parts of a codebase
                    // TODO: Needs to contain absolute decl file path...?
                    variable_description.push_str(
                        format!(", {}, decl {}:{}", &v.name, &v.decl_file, &v.decl_line,).as_str(),
                    );
                    // println!("{}", variable_description);

                    // Add unit name alongside inline ancestors to distinguish per-unit instances
                    let mut instance_segments = vec![function_stats.unit_name.clone()];
                    instance_segments.append(&mut inline_ancestors);

                    // TODO: if filtering_by_regions {
                    // Some paths are already absolute, others are relative to compilation
                    let mut decl_file_path_buf = if Path::new(&v.decl_dir).is_absolute() {
                        Path::new(&v.decl_dir).join(&v.decl_file)
                    } else {
                        Path::new(&function_stats.unit_dir)
                            .join(&v.decl_dir)
                            .join(&v.decl_file)
                    };
                    // Normalise on insert and access to allow for relative paths
                    decl_file_path_buf = decl_file_path_buf.absolutize().unwrap().to_path_buf();
                    let decl_file_path = decl_file_path_buf.to_str().unwrap();
                    // }

                    // Mark this variable as encountered
                    if let Some(ref mut encountered_variables) = encountered_variables_by_function {
                        let variable_set = encountered_variables
                            .entry((
                                leaf_function_name.clone(),
                                instance_segments.clone(),
                                decl_file_path.to_owned(),
                            ))
                            .or_default();
                        variable_set.insert(variable_description.clone());
                    }

                    if stats.opt.variables {
                        write!(
                            &mut w,
                            "{}\t{}",
                            &variable_description,
                            instance_segments.join(", "),
                        )?;
                    }

                    // TODO: Handle inlining in old metric or remove
                    let mut v_stats_adjusted = None;
                    if adjusting_by_baseline {
                        if let Some(bv) = base_v {
                            let mut source_line_set_adjusted =
                                v.extra.source_line_set_covered.clone();
                            if stats.opt.range_start_baseline {
                                // Baseline defines the start of coverage in source line terms
                                if let Some(start_line) = bv.extra.source_line_set_covered.first() {
                                    // let mut trimmed = false;
                                    while source_line_set_adjusted.first().unwrap_or(&u64::MAX)
                                        < start_line
                                    {
                                        source_line_set_adjusted.pop_first();
                                        // trimmed = true;
                                    }
                                    // if trimmed {
                                    //     print!("{}", &function_stats.name);
                                    //     println!(", {}, decl {}:{}, unit {}", &v.name, &v.decl_file, &v.decl_line, &function_stats.unit_name);
                                    //     println!("Start line: {}", start_line);
                                    //     println!("Base source line set: {:?}", bv.extra.source_line_set_covered);
                                    //     println!("Main source line set: {:?}", v.extra.source_line_set_covered);
                                    //     println!("Res  source line set: {:?}", source_line_set_adjusted);
                                    // }
                                } else {
                                    source_line_set_adjusted.clear();
                                }
                            }
                            if stats.opt.extend_from_baseline {
                                // Add additional range from end of baseline to end of scope
                                if let Some(source_line_set_after_covered) =
                                    &bv.extra.source_line_set_after_covered
                                {
                                    source_line_set_adjusted
                                        .append(&mut source_line_set_after_covered.clone());
                                    // if !source_line_set_after_covered.is_empty() {
                                    //     println!("After cov. source line set: {:?}", source_line_set_after_covered);
                                    //     println!("Main       source line set: {:?}", v.extra.source_line_set_covered);
                                    //     println!("Res        source line set: {:?}", source_line_set_adjusted);
                                    // }
                                }
                            }
                            v_stats_adjusted = Some(source_line_set_adjusted.len() as u64);
                        } else {
                            v_stats_adjusted = Some(0);
                        }
                    }

                    // Disable display of diff to baseline when using it to adjust main ranges
                    let bv_stats = if adjusting_by_baseline {
                        None
                    } else {
                        base_v.map(|bv| &bv.stats)
                    };

                    let mut v_stats_filtered = None;
                    let mut source_line_set_filtered = None;
                    let mut computation_line_set = None;
                    let mut first_defined_line = None;
                    if filtering_by_regions {
                        source_line_set_filtered = Some(v.extra.source_line_set_covered.clone());
                        if stats.opt.only_computation_regions {
                            let computation_line_sets_by_file =
                                computation_line_sets_by_file.as_ref().unwrap();
                            // if variable_description.starts_with("") {
                            //     println!("Unit dir: {}", function_stats.unit_dir);
                            //     println!("Decl dir: {}", v.decl_dir);
                            //     println!("Decl file: {}", v.decl_file);
                            //     println!("Decl file path: {}", decl_file_path.to_str().unwrap());
                            // }
                            computation_line_set =
                                computation_line_sets_by_file.get(decl_file_path);
                            if let Some(computation_line_set) = computation_line_set {
                                source_line_set_filtered = source_line_set_filtered.map(|set| {
                                    set.intersection(computation_line_set).cloned().collect()
                                });
                            } else {
                                source_line_set_filtered.as_mut().map(|set| set.clear());
                            }
                        }
                        if stats.opt.range_start_first_defined_region {
                            let first_defined_line_by_variable =
                                first_defined_line_by_variable.as_ref().unwrap();
                            first_defined_line =
                                first_defined_line_by_variable.get(&variable_description);
                            if let Some(first_defined_line) = first_defined_line {
                                source_line_set_filtered.as_mut().map(|set| {
                                    while set.first().unwrap_or(&u64::MAX) < first_defined_line {
                                        set.pop_first();
                                    }
                                });
                            } else {
                                source_line_set_filtered.as_mut().map(|set| {
                                    set.clear();
                                });
                            }
                        }
                        if stats.opt.only_cachegrind_regions {
                            let cachegrind_line_sets_by_file =
                                cachegrind_line_sets_by_file.as_ref().unwrap();
                            cachegrind_line_set = cachegrind_line_sets_by_file.get(decl_file_path);
                            if let Some(cachegrind_line_set) = cachegrind_line_set {
                                source_line_set_filtered = source_line_set_filtered.map(|set| {
                                    set.intersection(cachegrind_line_set).cloned().collect()
                                });
                            } else {
                                source_line_set_filtered.as_mut().map(|set| set.clear());
                            }
                        }
                        v_stats_filtered = source_line_set_filtered
                            .as_ref()
                            .map(|set| set.len() as u64);
                    }

                    let mut src_scope_lines = None;
                    let mut scope_line_set = None;
                    if stats.opt.scope_regions {
                        scope_line_set = scope_line_set_for_variable(
                            &stats.opt,
                            &scope_line_sets_by_variable,
                            &variable_description,
                            computation_line_set,
                            first_defined_line,
                        );
                        src_scope_lines =
                            Some(scope_line_set.as_ref().map_or(0, |set| set.len() as u64));
                    }

                    if filtering_by_regions && stats.opt.scope_regions {
                        if let Some(scope_line_set) = scope_line_set.as_ref() {
                            source_line_set_filtered = source_line_set_filtered
                                .map(|set| set.intersection(scope_line_set).cloned().collect());

                            // If knowledge extension is enabled, look for any trailing lines in
                            // the scope line set that can be added.
                            if stats.opt.extend_from_baseline {
                                let last_covered_line =
                                    source_line_set_filtered.as_ref().and_then(|set| set.last());
                                if let Some(last_covered_line) = last_covered_line {
                                    let mut after_covered = scope_line_set
                                        .range((last_covered_line + 1)..)
                                        .cloned()
                                        .collect();
                                    source_line_set_filtered
                                        .as_mut()
                                        .map(|set| set.append(&mut after_covered));
                                }
                            }

                            v_stats_filtered = source_line_set_filtered
                                .as_ref()
                                .map(|set| set.len() as u64);
                        }
                        // We could clear filtered coverage when scope lines are missing,
                        // but presumably the NaN from N / 0 already suggests a problem.
                    }

                    // if variable_description.starts_with("") {
                    //     println!("Variable: {}", variable_description);
                    //     println!("Covered line set: {:?}", v.extra.source_line_set_covered);
                    //     println!("Computation line set: {:?}", computation_line_set);
                    //     println!("First defined line: {:?}", first_defined_line);
                    //     println!("Cachegrind line set: {:?}", cachegrind_line_set);
                    //     println!("Scope line set: {:?}", scope_line_set);
                    //     println!("Filtered line set: {:?}", source_line_set_filtered);
                    // }

                    if let Some(ref mut locatable_vars_by_line) = locatable_vars_by_line {
                        // Use filtered set if it exists, otherwise fallback to basic coverage
                        let covered_line_set = source_line_set_filtered
                            .as_ref()
                            .unwrap_or(&v.extra.source_line_set_covered);
                        for line in covered_line_set {
                            let locatable_vars = locatable_vars_by_line.entry(*line).or_default();
                            locatable_vars.insert(variable_description.clone());
                        }
                    }
                    if let Some(ref mut scope_vars_by_line) = scope_vars_by_line {
                        let scope_line_set = scope_line_set.as_ref().unwrap();
                        for line in scope_line_set {
                            let scope_vars = scope_vars_by_line.entry(*line).or_default();
                            scope_vars.insert(variable_description.clone());
                        }
                    }

                    if stats.opt.variables {
                        write_stats(
                            &mut w,
                            &v.stats,
                            bv_stats,
                            v_stats_adjusted,
                            v_stats_filtered,
                            src_scope_lines,
                        );
                    }
                }

                // Look for any variables in source analysis that we did not encounter
                if let Some(ref encountered_variables_bf) = encountered_variables_by_function {
                    for key in encountered_variables_bf.keys() {
                        let (leaf_function_name, instance_segments, decl_file_path) = key;
                        let encountered_variables = encountered_variables_bf.get(key).unwrap();
                        // Borrow the declaration file from the first variable
                        let first_variable = encountered_variables.iter().next().unwrap();
                        let decl_file = first_variable
                            .split(", ")
                            .nth(2)
                            .unwrap()
                            .split(":")
                            .next()
                            .unwrap()
                            .to_owned();
                        // TODO: Figure out some kind of file for such cases
                        if decl_file.contains("<unknown file>") {
                            continue;
                        }
                        let expected_variables = variables_by_function
                            .as_ref()
                            .unwrap()
                            .get(&(leaf_function_name.clone(), decl_file.clone()));
                        // TODO: Work out what's happening here (see `tar`)
                        if expected_variables.is_none() {
                            // println!(
                            //     "No expected variables for {}, {}",
                            //     leaf_function_name, &decl_file,
                            // );
                            continue;
                        }
                        let missing_variables = expected_variables
                            .unwrap()
                            .difference(encountered_variables);
                        for variable_description in missing_variables {
                            if stats.opt.only_locals {
                                let var_name = variable_description.split(", ").nth(1).unwrap();
                                // Borrow ASPLOS name filter for now
                                // x.startswith('l') or x in ['i', 'j', 'k', 'print_hash_value']
                                // TODO: Emit local vs. parameter in source analysis
                                if !var_name.starts_with("l")
                                    && var_name != "i"
                                    && var_name != "j"
                                    && var_name != "k"
                                    && var_name != "print_hash_value"
                                {
                                    continue;
                                }
                            }
                            let mut computation_line_set = None;
                            let mut first_defined_line = None;
                            if stats.opt.only_computation_regions {
                                let computation_line_sets_by_file =
                                    computation_line_sets_by_file.as_ref().unwrap();
                                computation_line_set =
                                    computation_line_sets_by_file.get(decl_file_path);
                            }
                            if stats.opt.range_start_first_defined_region {
                                let first_defined_line_by_variable =
                                    first_defined_line_by_variable.as_ref().unwrap();
                                first_defined_line =
                                    first_defined_line_by_variable.get(variable_description);
                            }
                            let scope_line_set = scope_line_set_for_variable(
                                &stats.opt,
                                &scope_line_sets_by_variable,
                                &variable_description,
                                computation_line_set,
                                first_defined_line,
                            );
                            let src_scope_lines =
                                scope_line_set.as_ref().map_or(0, |set| set.len() as u64);
                            // println!(
                            //     "Missing variable: {}, instance: {}, lines: {}",
                            //     variable_description,
                            //     instance_segments.join(", "),
                            //     &src_scope_lines,
                            // );
                            if stats.opt.variables {
                                write!(
                                    &mut w,
                                    "{}\t{}",
                                    &variable_description,
                                    instance_segments.join(", "),
                                )?;
                                write_stats(
                                    &mut w,
                                    &VariableStats {
                                        instruction_bytes_in_scope: 0,
                                        instruction_bytes_covered: 0,
                                        source_lines_in_scope: 0,
                                        source_lines_covered: 0,
                                    },
                                    None,
                                    if adjusting_by_baseline { Some(0) } else { None },
                                    if filtering_by_regions { Some(0) } else { None },
                                    Some(src_scope_lines),
                                );
                            }
                            if let Some(ref mut scope_vars_by_line) = scope_vars_by_line {
                                let scope_line_set = scope_line_set.as_ref().unwrap();
                                for line in scope_line_set {
                                    let scope_vars = scope_vars_by_line.entry(*line).or_default();
                                    scope_vars.insert(variable_description.clone());
                                }
                            }
                        }
                    }
                }
            } else {
                write!(
                    &mut w,
                    "{}, unit {}",
                    &function_stats.name, &function_stats.unit_name
                )?;
                write_stats(
                    &mut w,
                    &function_stats.stats,
                    base_function_stats.map(|b| &b.stats),
                    None,
                    None,
                    None,
                );
            }
        }

        if stats.opt.lines {
            // TODO: Support multiple source files in line mode
            let mut computation_line_set = None;
            let computation_line_sets_by_file = computation_line_sets_by_file.as_ref().unwrap();
            // Filter to file for ASPLOS test cases
            for file in computation_line_sets_by_file.keys() {
                if !file.ends_with("a.c") {
                    continue;
                }
                computation_line_set = computation_line_sets_by_file.get(file);
            }

            // Gather all lines for local variables found via source analysis
            // Can't depend on DWARF here, as some variable may not be emitted
            let mut report_scope_lines = BTreeSet::new();
            // TODO: Check options appropriately
            let scope_line_sets_by_variable = scope_line_sets_by_variable.as_ref().unwrap();
            let first_defined_line_by_variable = first_defined_line_by_variable.as_ref().unwrap();
            for variable in scope_line_sets_by_variable.keys() {
                let mut variable_parts = variable.split(", ");
                let function = variable_parts.next().unwrap();

                // ASPLOS considers only functions with `main` or starting with `func`
                if !function.contains("main") && !function.starts_with("func") {
                    continue;
                }

                let mut variable_scope_lines =
                    scope_line_sets_by_variable.get(variable).unwrap().clone();
                variable_scope_lines = variable_scope_lines
                    .intersection(computation_line_set.unwrap())
                    .cloned()
                    .collect();
                let first_defined_line = first_defined_line_by_variable.get(variable);
                if let Some(first_defined_line) = first_defined_line {
                    while variable_scope_lines.first().unwrap_or(&u64::MAX) < first_defined_line {
                        variable_scope_lines.pop_first();
                    }
                } else {
                    variable_scope_lines.clear();
                }
                report_scope_lines.append(&mut variable_scope_lines);
            }

            let lines_present = stats.lines.unwrap();
            let lines_reachable = cachegrind_line_set;
            let locatable_vars_by_line = locatable_vars_by_line.unwrap();
            let scope_vars_by_line = scope_vars_by_line.unwrap();

            // Iterate through source analysis lines gathered above
            for ref line in report_scope_lines {
                let line_reachable = lines_reachable.map_or(true, |lr| lr.contains(line));
                let locatable_vars = locatable_vars_by_line.get(line);
                let scope_vars = scope_vars_by_line.get(line);
                writeln!(
                    &mut w,
                    "{:12}\t{:12}\t{:12}\t{:12}\t{:12}",
                    // "{:12}\t{:12}\t{:12}\t{:12}: {:?}\t{:12}: {:?}",
                    line,
                    if lines_present.contains(line) { 1 } else { 0 },
                    if line_reachable { 1 } else { 0 },
                    locatable_vars.map(|v| v.len()).unwrap_or_default(),
                    // locatable_vars,
                    scope_vars.map(|v| v.len()).unwrap_or_default(),
                    // scope_vars,
                )?;
            }

            if !stats.opt.tsv {
                writeln!(&mut w)?;
                writeln!(
                    &mut w,
                    "Lines with locatable variables: {}",
                    locatable_vars_by_line.keys().len()
                )?;
                writeln!(
                    &mut w,
                    "Lines with scope variables: {}",
                    scope_vars_by_line.keys().len()
                )?;
            }
        }

        writeln!(&mut w)?;
    }

    if adjusting_by_baseline {
        base_stats = None;
    }
    // When adjusting by baseline, we currently don't recompute accumulated stats, so disable them
    // to avoid misleading the user.
    // In TSV mode, we assume some further tooling will analyse the data, so we also disable
    // accumulated output.
    if !adjusting_by_baseline && !stats.opt.tsv {
        if !stats.opt.only_locals {
            write_stats_label(
                &mut w,
                "params",
                &stats.bundle.parameters,
                base_stats.as_ref().map(|b| &b.bundle.parameters),
                &stats.opt,
            );
        }
        if !stats.opt.only_parameters {
            write_stats_label(
                &mut w,
                "vars",
                &stats.bundle.variables,
                base_stats.as_ref().map(|b| &b.bundle.variables),
                &stats.opt,
            );
        }
        if !stats.opt.only_locals && !stats.opt.only_parameters {
            let all = stats.bundle.variables + stats.bundle.parameters;
            let base_all = base_stats
                .as_ref()
                .map(|b| b.bundle.variables.clone() + b.bundle.parameters.clone());
            write_stats_label(&mut w, "all", &all, base_all.as_ref(), &stats.opt);
        }
    }

    Ok(())
}
