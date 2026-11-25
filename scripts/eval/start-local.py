#!/usr/bin/env python3
"""Run an isolated evaluation catalog and Elasticsearch index on the host."""
import os, pathlib, shlex, subprocess, hashlib, shutil
root=pathlib.Path(__file__).resolve().parents[2]
env=os.environ.copy()
settings = root / '.env'
for line in settings.read_text().splitlines() if settings.exists() else ():
    if not line.strip() or line.lstrip().startswith('#') or '=' not in line: continue
    key,value=line.split('=',1)
    parsed=shlex.split(value,comments=True)
    env.setdefault(key.strip(),parsed[0] if parsed else '')
env.update({'EVIDENCE_RAG_ES_INDEX': 'evidence_rag_eval_v1',
            'EVIDENCE_RAG_DATA_DIR': str(root / 'data/evidence-rag-eval'),
            'EVIDENCE_RAG_QUERY_CACHE_ENTRIES': '1024',
            'EVIDENCE_RAG_HOST': '127.0.0.1',
            'EVIDENCE_RAG_PORT': env.get('EVIDENCE_RAG_EVAL_HTTP_PORT', '18081')})
java = env.get('EVIDENCE_RAG_JAVA')
if not java and pathlib.Path('/usr/libexec/java_home').exists():
    java = subprocess.check_output(['/usr/libexec/java_home', '-v', '21'], text=True).strip() + '/bin/java'
java = java or shutil.which('java')
if not java:
    raise SystemExit('Java 21 is required')
artifact = root / 'target/evidence-rag-java.jar'
snapshot = root / 'run' / ('evidence-rag-' + hashlib.sha256(artifact.read_bytes()).hexdigest() + '.jar')
snapshot.parent.mkdir(exist_ok=True)
if not snapshot.exists(): shutil.copy2(artifact,snapshot)
os.chdir(root)
os.execve(java,[java,'-Xmx1200m','-jar',str(snapshot)],env)
