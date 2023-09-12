extern crate object;
extern crate structopt;
extern crate typed_arena;

use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::iter::Iterator;
use std::path::Path;
use std::process;

use debuginfo_quality::*;
use object::{Object, ObjectSection};
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
    adj_stats: Option<&VariableStatsAdjustment>,
) {
    if let Some(b) = base_stats {
        writeln!(w,
                 "\t{:12}\t{:12}\t{:12}\t{:12}\
                  \t{:12}\t{:12}\t{:12}\t{:12}",
                 stats.instruction_bytes_covered,
                 stats.instruction_bytes_in_scope,
                 b.instruction_bytes_covered,
                 b.instruction_bytes_in_scope,
                 stats.source_lines_covered,
                 stats.source_lines_in_scope,
                 b.source_lines_covered,
                 b.source_lines_in_scope).unwrap();
    } else if let Some(adj) = adj_stats {
        writeln!(w, "\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}",
                 stats.instruction_bytes_covered,
                 stats.instruction_bytes_in_scope,
                 stats.source_lines_covered,
                 adj.source_lines_covered_adjusted,
                 stats.source_lines_in_scope).unwrap();
    } else {
        writeln!(w, "\t{:12}\t{:12}\t{:12}\t{:12}",
                 stats.instruction_bytes_covered,
                 stats.instruction_bytes_in_scope,
                 stats.source_lines_covered,
                 stats.source_lines_in_scope).unwrap();
    }
}

fn write_stats_label<W: io::Write>(mut w: W, label: &str, stats: &VariableStats, base_stats: Option<&VariableStats>, opt: &Opt) {
    write!(w, "{}", label).unwrap();
    if !opt.tsv && (opt.functions || opt.variables) {
        write!(&mut w, "\t\t\t\t").unwrap();
    }
    write_stats(w, stats, base_stats, None);
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
        'a: 'file
    {
        let data = match file.section_by_name(S::section_name()) {
            Some(ref section) => section.uncompressed_data().unwrap_or(Cow::Borrowed(&[][..])),
            None => Cow::Borrowed(&[][..]),
        };
        let data_ref = (*arena.alloc(data)).borrow();
        S::from(gimli::EndianSlice::new(data_ref, gimli::LittleEndian))
    }

    let mut stats = Stats { bundle: StatsBundle::default(), opt: opt.clone(), output: Vec::new() };
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

        evaluate_info(debug_info, debug_abbrev, debug_str, rnglists, loclists, debug_line,
                      stats.opt.no_entry_value, stats.opt.no_parameter_ref, &mut stats);
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

        let mut stats = Stats { bundle: StatsBundle::default(), opt: opt.clone(), output: Vec::new() };
        evaluate_info(debug_info, debug_abbrev, debug_str, rnglists, loclists, debug_line,
                      stats.opt.no_entry_value_baseline, stats.opt.no_parameter_ref_baseline, &mut stats);
        base_stats = Some(stats);
    }

    let stdout = io::stdout();
    let mut stdout_locked = stdout.lock();
    let mut w = BufWriter::new(&mut stdout_locked);

    if !stats.opt.tsv && (stats.opt.functions || stats.opt.variables) {
        write!(&mut w, "\t\t\t\t")?;
    }
    write!(&mut w, "Name")?;
    let adjusting_by_baseline = stats.opt.range_start_baseline || stats.opt.extend_from_baseline;
    if base_stats.is_some() && !adjusting_by_baseline {
        writeln!(&mut w,
                 "\t{:12}\t{:12}\t{:12}\t{:12}\
                  \t{:12}\t{:12}\t{:12}\t{:12}",
                 "Cov (B)", "Scope (B)", "BaseCov (B)", "BaseScope (B)",
                 "Cov (L)", "Scope (L)", "BaseCov (L)", "BaseScope (L)")?;
    } else if adjusting_by_baseline {
        writeln!(&mut w, "\t{:12}\t{:12}\t{:12}\t{:12}\t{:12}",
                 "Cov (B)", "Scope (B)",
                 "Raw Cov (L)", "Adj Cov (L)", "Scope (L)")?;
    } else {
        writeln!(&mut w, "\t{:12}\t{:12}\t{:12}\t{:12}",
                 "Cov (B)", "Scope (B)",
                 "Cov (L)", "Scope (L)")?;
    }
    writeln!(&mut w)?;
    if stats.opt.functions || stats.opt.variables {
        let mut functions: Vec<(FunctionStats, Option<&FunctionStats>)> = if let Some(base) = base_stats.as_ref() {
            let mut base_functions = HashMap::new();
            // TODO: Avoid cloning keys here...?
            for f in base.output.iter() {
                base_functions.insert((f.name.clone(), f.unit_name.clone()), f);
            }
            stats.output.into_iter().filter_map(|o|
                base_functions.get(&(o.name.clone(), o.unit_name.clone())).map(|b| (o, Some(*b)))
            ).collect()
        } else {
            stats.output.into_iter().map(|o| (o, None)).collect()
        };
        functions.sort_by(|a, b| goodness(a).partial_cmp(&goodness(b)).unwrap());
        for (function_stats, base_function_stats) in functions {
            if stats.opt.variables {
                let unit_name = function_stats.unit_name.clone();
                for v in function_stats.variables {
                    let base_v = base_function_stats.and_then(|bf| {
                        let mut same_v = bf.variables.iter().filter(|&bv|
                            bv.name == v.name &&
                            bv.decl_file == v.decl_file &&
                            bv.decl_line == v.decl_line &&
                            bf.unit_name == unit_name
                        );
                        if same_v.clone().count() == 1 {
                            same_v.next()
                        } else {
                            None
                        }
                    });
                    write!(&mut w, "{}", &function_stats.name)?;
                    for inline in v.inlines {
                        write!(&mut w, ", {}", &inline)?;
                    }
                    write!(&mut w, ", {}, decl {}:{}, unit {}", &v.name, &v.decl_file, &v.decl_line, &function_stats.unit_name)?;
                    let mut v_stats_adjustment = None;
                    if adjusting_by_baseline {
                        if let Some(bv) = base_v {
                            let mut source_line_set_adjusted = v.extra.source_line_set_covered.clone();
                            if stats.opt.range_start_baseline {
                                // Baseline defines the start of coverage in source line terms
                                if let Some(start_line) = bv.extra.source_line_set_covered.first() {
                                    // let mut trimmed = false;
                                    while source_line_set_adjusted.first().unwrap_or(&u64::MAX) < start_line {
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
                                if let Some(source_line_set_after_covered) = &bv.extra.source_line_set_after_covered {
                                    source_line_set_adjusted.append(&mut source_line_set_after_covered.clone());
                                    // if !source_line_set_after_covered.is_empty() {
                                    //     println!("After cov. source line set: {:?}", source_line_set_after_covered);
                                    //     println!("Main       source line set: {:?}", v.extra.source_line_set_covered);
                                    //     println!("Res        source line set: {:?}", source_line_set_adjusted);
                                    // }
                                }
                            }
                            v_stats_adjustment = Some(VariableStatsAdjustment {
                                source_lines_covered_adjusted: source_line_set_adjusted.len() as u64,
                            });
                        }
                    }
                    // Disable display of diff to baseline when using it to adjust main ranges
                    let bv_stats = if adjusting_by_baseline {
                        None
                    } else {
                        base_v.map(|bv| &bv.stats)
                    };
                    write_stats(&mut w, &v.stats, bv_stats, v_stats_adjustment.as_ref());
                }
            } else {
                write!(&mut w, "{}, unit {}", &function_stats.name, &function_stats.unit_name)?;
                write_stats(&mut w, &function_stats.stats, base_function_stats.map(|b| &b.stats), None);
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
            write_stats_label(&mut w, "params", &stats.bundle.parameters, base_stats.as_ref().map(|b| &b.bundle.parameters), &stats.opt);
        }
        if !stats.opt.only_parameters {
            write_stats_label(&mut w, "vars", &stats.bundle.variables, base_stats.as_ref().map(|b| &b.bundle.variables), &stats.opt);
        }
        if !stats.opt.only_locals && !stats.opt.only_parameters {
            let all = stats.bundle.variables + stats.bundle.parameters;
            let base_all = base_stats.as_ref().map(|b| b.bundle.variables.clone() + b.bundle.parameters.clone());
            write_stats_label(&mut w, "all", &all, base_all.as_ref(), &stats.opt);
        }
    }
    Ok(())
}

fn goodness(&(ref a, ref a_base): &(FunctionStats, Option<&FunctionStats>)) -> (f64, i64) {
    (if let Some(a_base) = a_base.as_ref() {
        a.stats.fraction_bytes_covered() - a_base.stats.fraction_bytes_covered()
    } else {
        a.stats.fraction_bytes_covered()
    }, -(a.stats.instruction_bytes_in_scope as i64))
}
