# Explicitly authorized resume of the isolated window-step project.
# No old overlay/package is replaced; physical timing gates bit generation.
if {$argc != 1} { error "Usage: resume_lung_window_step.tcl EXPERIMENT_DIR" }
set task_root [file normalize [file join [file dirname [info script]] ..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0} { error "Use workspace/data experiment" }
set project_path [file join $task_out project lung_network.xpr]
if {![file exists $project_path] || [file exists [file join $task_out verification.json]] ||
    [file exists [file join $task_out deploy lung_network.bit]]} {
    error "Need unfinished experiment; never overwrite accepted proof or bit"
}
open_project $project_path
if {[get_property PART [current_project]] ne "xc7z020clg400-1"} { error "Wrong FPGA part" }
open_bd_design [get_files design_1.bd]
source [file join $task_root fpga validate_lung_window_step.tcl]
update_compile_order -fileset sources_1
set prior_logs [file join $task_out interrupted_logs]
if {[file exists $prior_logs]} { error "Preserve prior resume attempt; inspect before retry" }
file mkdir $prior_logs
foreach r [get_runs] {
    puts "Prior run $r: [get_property STATUS $r]; [get_property PROGRESS $r]"
    set runlog [file join [get_property DIRECTORY $r] runme.log]
    if {[file exists $runlog]} { file copy $runlog [file join $prior_logs "${r}_runme.log"] }
}
# The external process tree was verified stopped before entering this script.
# Reset only unfinished runs, not successful prerequisite IP synthesis results.
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    reset_runs [get_runs synth_1]
    foreach r [get_runs *_synth_1] {
        if {[get_property PROGRESS $r] ne "100%"} { reset_runs $r }
    }
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
}
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "Resumed synthesis failed" }
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {![string match "*route_design Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Implementation did not finish route_design"
}
open_run impl_1
report_utilization -file [file join $task_out utilization_routed.rpt]
report_timing_summary -delay_type min_max -max_paths 10 -file [file join $task_out timing_routed.rpt]
report_drc -file [file join $task_out drc.rpt]
report_exceptions -file [file join $task_out configuration_exceptions.rpt]
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup] == 0 || [get_property SLACK $setup] < 0 ||
    [llength $hold] == 0 || [get_property SLACK $hold] < 0} {
    error "Routed setup/hold failed; no bitstream generation"
}
foreach v [get_drc_violations -quiet] {
    if {[get_property SEVERITY $v] in {Error {Critical Warning}}} { error "DRC failed: $v" }
}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "Bitstream generation failed" }
file mkdir [file join $task_out deploy]
file copy [file join $task_out project lung_network.runs impl_1 design_1_wrapper.bit] [file join $task_out deploy lung_network.bit]
file copy [file join $task_out project lung_network.gen sources_1 bd design_1 hw_handoff design_1.hwh] [file join $task_out deploy lung_network.hwh]
puts "Resumed route and timing-gated bit generation complete; independent package audit and board test still required."
close_project
