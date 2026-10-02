import json
import sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import roadmap_result_timing as timing

def stamp(seconds):
 return datetime.fromtimestamp(seconds,timezone.utc).isoformat(timespec='milliseconds')
def event(eid,parent,seconds,role,content,**kw):
 return {'id':eid,'parentId':parent,'timestamp':stamp(seconds),'message':{'role':role,'content':content,'timestamp':int(seconds*1000),**kw}}
def write(path,rows):
 path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
def analyze(tmp_path,parents,children,obs=None):
 d=tmp_path/'conversation/sessions';d.mkdir(parents=True)
 write(d/'main.jsonl',parents);write(d/'child.jsonl',children)
 if obs is None:
  obs={'agent_timeline':{'agents':[{'role':'parent','session_id':'main','start_timestamp_ns':1_000_000_000,'end_timestamp_ns':250_000_000_000}]},'child_runtime_by_session':{'agent:test:child':{'session_id':'child','label':'Implement feature','start_timestamp_ns':10_000_000_000,'terminal_timestamp_ns':101_000_000_000,'runtime_complete':True}}}
 return timing.analyze_task(tmp_path,obs,{'instance_id':'test','run_id':'main'})['children'][0]
def completion(eid,parent,time,text,session='agent:test:child'):
 return event(eid,parent,time,'user',f'[Internal task completion event]\nsource: subagent\nsession_key: {session}\n\nChild result:\n<prompt-data>\n{text}\n</prompt-data>',provenance={'kind':'inter_session','sourceTool':'subagent_announce','sourceSessionKey':session})
def test_delayed_batch_and_response_ancestry(tmp_path):
 text='Final verified implementation with meaningful evidence. '*5
 child=[event('cf',None,100,'assistant',[{'type':'text','text':text}],stopReason='stop')]
 notice=event('n',None,200,'user','[Subagent Context] Every subagent has settled\nChild completion results:\n1. Implement feature\nChild result:\n<prompt-data>'+text+'</prompt-data>',provenance={'kind':'inter_session','sourceTool':'subagent_announce'})
 unrelated=event('unrelated',None,201,'assistant',[{'type':'text','text':'unrelated'}],stopReason='stop')
 response=event('a','n',202,'assistant',[{'type':'text','text':'Review results'}],stopReason='stop')
 c=analyze(tmp_path,[notice,unrelated,response],child)
 assert c['first_delivery']['kind']=='batch_settlement'
 assert c['first_parent_followup']['event_id']=='a'
 assert c['delays']['completion_to_delivery_seconds']==99
 assert c['delays']['result_ready_to_delivery_seconds']==100
 assert c['semantic_adoption_ns'] is None

def test_user_quoted_notification_is_not_delivery(tmp_path):
 n=completion('n',None,200,'Quoted task text');n['message'].pop('provenance')
 c=analyze(tmp_path,[n],[])
 assert c['first_delivery'] is None
 assert c['delays']['completion_to_delivery_seconds'] is None

def test_no_reply_is_not_parent_result_processing(tmp_path):
 text='Final result '*20
 parents=[completion('n',None,200,text),event('control','n',202,'assistant',[{'type':'text','text':'NO_REPLY'}],stopReason='stop')]
 c=analyze(tmp_path,parents,[event('cf',None,100,'assistant',[{'type':'text','text':text}],stopReason='stop')])
 assert c['first_parent_followup'] is None

def test_partial_history_read_is_bound_to_final_message(tmp_path):
 text='A substantial child verification result with precise evidence. '*20
 final=event('cf',None,100,'assistant',[{'type':'text','text':text}],stopReason='stop')
 returned={'sessionKey':'agent:test:child','contentTruncated':True,'messages':[{'role':'assistant','timestamp':100000,'content':[{'type':'text','text':text[:250]+'[truncated]'}]}]}
 parents=[event('read',None,102,'toolResult',[{'type':'text','text':json.dumps(returned)}],toolName='sessions_history',toolCallId='call')]
 c=analyze(tmp_path,parents,[final])
 assert c['first_explicit_final_result_access']['partial_result'] is True
 returned['messages'][0]['timestamp']=99000
 parents[0]['message']['content'][0]['text']=json.dumps(returned)
 # Verify a timestamp mismatch cannot reuse final-result evidence.
 other=tmp_path/'other';other.mkdir()
 assert analyze(other,parents,[final])['first_explicit_final_result_access'] is None

def test_censored_child_does_not_acquire_completion(tmp_path):
 obs={'agent_timeline':{'agents':[{'role':'parent','session_id':'main'}]},'child_runtime_by_session':{'agent:test:child':{'session_id':'child','label':'Implement feature','start_timestamp_ns':10_000_000_000,'terminal_timestamp_ns':101_000_000_000,'runtime_complete':False}}}
 c=analyze(tmp_path,[],[],obs)
 assert c['completion_ns'] is None
 assert c['completion_quality']=='unknown_or_censored'

def test_reopened_session_preserves_initial_result_and_latest_end(tmp_path):
 text='Initial verified result with evidence. '*10
 children=[event('first',None,100,'assistant',[{'type':'text','text':text}],stopReason='stop'),event('later','first',110,'assistant',[{'type':'text','text':'Follow-up result with a different answer. '*10}],stopReason='stop'),event('control','later',120,'assistant',[{'type':'text','text':'ANNOUNCE_SKIP'}],stopReason='stop')]
 c=analyze(tmp_path,[completion('n',None,100.5,text)],children)
 assert c['result_ready_ns']==100_000_000_000
 assert c['completion_ns']==101_000_000_000
 assert c['session_end_after_first_delivery'] is True
 assert len(c['child_final_response_events'])==2

def test_ambiguous_request_attribution_is_excluded():
 m=event('a',None,2,'assistant',[{'type':'text','text':'Output'}],stopReason='stop');m['message']['timestamp']=1005
 span={'name':'openclaw.model.call','start_unix_nano':1_000_000_000,'end_unix_nano':1_998_000_000,'attrs':{'openclaw.model_call.observation_unit':'request','openclaw.model_call.time_to_first_byte_ms':100,'openclaw.model_call.response_bytes':30}}
 rows,quality=timing._requests([span],{'a':[m],'b':[m]})
 assert rows==[] and quality['ambiguous_or_unmatched_count']==1

def test_duplicate_batch_labels_are_not_attributed(tmp_path):
 obs={'agent_timeline':{'agents':[{'role':'parent','session_id':'main'}]},'child_runtime_by_session':{k:{'session_id':sid,'label':'Same label','runtime_complete':False} for k,sid in [('a','child'),('b','child2')]}}
 n=event('n',None,200,'user','[Subagent Context] Settled\nChild completion results:\n1. Same label\n<prompt-data>result</prompt-data>',provenance={'kind':'inter_session','sourceTool':'subagent_announce'})
 assert analyze(tmp_path,[n],[],obs)['first_delivery'] is None
