"""One metadata-only process regression for surviving a parent handoff."""
import json,pathlib,subprocess,sys,tempfile,time
from worker_service import keep_step_alive

root=pathlib.Path(tempfile.mkdtemp(prefix='lava_service_fixture_'))
program=r'''
import json,pathlib,subprocess,sys,time
from worker_service import keep_step_alive
r=pathlib.Path(sys.argv[1])
# Stand-in for an original worker that exits while a successor still runs.
successor="import pathlib,time; p=pathlib.Path("+repr(str(r))+ "); time.sleep(.7); (p/'successor_finished').write_text('yes'); (p/'EXECUTION_OUTCOMES_COMPLETE.json').write_text('{}')"
old="import subprocess,sys; subprocess.Popen([sys.executable,'-c',"+repr(successor)+"],start_new_session=True)"
p=subprocess.Popen([sys.executable,'-c',old]);p.wait(timeout=3)
(r/'original_exited').write_text('yes')
keep_step_alive(r,time.time()+5,[p],interval=.02)
(r/'launcher_exited').write_text('yes')
'''
p=subprocess.Popen([sys.executable,'-c',program,str(root)])
end=time.monotonic()+3
while not(root/'original_exited').exists()and time.monotonic()<end:time.sleep(.01)
assert(root/'original_exited').exists()
assert p.poll()is None and not(root/'launcher_exited').exists()
assert p.wait(timeout=4)==0 and(root/'successor_finished').exists()
deadline=time.time()+.05;keep_step_alive(root/'separate_cutoff',deadline,interval=.01)
assert time.time()>=deadline
print(json.dumps({'status':'passed','original_PID_exit_did_not_end_step_service':True,
 'successor_survived_until_terminal_marker':True,'finite_cutoff_checked':True,
 'scientific_fits':0,'solves':0,'enumerations':0,'fixture':str(root)}))
