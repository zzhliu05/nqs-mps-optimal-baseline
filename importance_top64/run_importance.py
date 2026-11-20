"""Portable entry point for the validated importance-sampling baseline."""
import argparse,hashlib,json,os,sys
from pathlib import Path

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--config',default=str(root/'config.json'))
parser.add_argument('--output',required=True)
parser.add_argument('--steps',type=int)
parser.add_argument('--tk',type=int)
parser.add_argument('--top-exact',type=int,choices=(32,64),default=64)
parser.add_argument('--exact',action='store_true',help='Full Hamiltonian neighbor summation')
parser.add_argument('--no-mask',action='store_true',help='Original all-proposal MCMC forward')
parser.add_argument('--resume')
parser.add_argument('--resume-sector',type=int)
args=parser.parse_args()
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
config=json.loads(Path(args.config).read_text())
if args.steps is not None:config['training']['steps']=args.steps
if args.tk is not None:config['system']['tk']=args.tk
if config['system']['nphi']!=51 or config['network']['bond_dim']!=10:
    raise ValueError('Frozen baseline validated only for Nphi51/D10; adapt and revalidate other systems')
if args.exact:
    config['execution']['chain_batch']=min(4096,config['execution']['chain_batch'])
    slots=config['execution']['chain_batch']*config['execution']['row_width']
    config['execution']['dedup_chunks']=max(config['execution']['dedup_chunks'],(slots+32767)//32768)
os.environ['NUMBA_NUM_THREADS']=str(config['execution']['cpu_threads'])
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ['IS_FUSED_CPU_PACK']='1'
os.environ['IS_PROPOSAL_PARALLEL']='1'
os.environ['IS_TOP_EXACT']=str(args.top_exact)
sys.path[:0]=[str(root/'src'),str(root/'implementation')]
from nqs_mps import cli
original=cli.configure


def configure(settings):
    result=original(settings)
    import numpy as np
    from nqs_torus_laqx_nes import training
    if not args.no_mask:
        from masked_sampler import install
        install(out,1024)
    if not args.exact:
        from cpu_sample_pack import sample_pack
        if os.environ.get('NQS_COMPACT_STAGE_CPU')!='1':raise ValueError('compact CPU staging required')
        capacity=config['execution']['chain_batch']*(args.top_exact+32)
        os.environ['NQS_COMPACT_DEDUP_SIZE']=str(capacity)
        rng=np.random.default_rng(912039)
        def pack(rows,unused_capacity,value_dtype=np.complex128):
            packed,timing=sample_pack(rows,rng,value_dtype)
            if len(packed[0])>capacity:raise ValueError('sample buffer overflow')
            with (out/'cpu_sample_pack.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
            return packed
        training.pack_sparse_bitstring_rows=pack
    return result


cli.configure=configure
resolved=out/'input_config.json';resolved.write_text(json.dumps(config,indent=2))
manifest={'importance_sampling':not args.exact,'top_exact':args.top_exact,'tail_draws':32,
          'proposal':'0.9*abs(H)/sum_tail_abs(H)+0.1/n_tail','proposal_seed':912039,
          'fused_cpu_pack':not args.exact,'gpu_dedup':True,'state_cache':False,
          'masked_sampler':not args.no_mask,'sampler_tile':1024,
          'warning':'runtime resolved/history exact label describes underlying evaluator; consult this manifest for actual estimator',
          'validation':'short performance/equality tests only; long convergence not certified',
          'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob('*.py') if '__pycache__' not in str(p)}}
(out/'importance_manifest.json').write_text(json.dumps(manifest,indent=2))
sys.argv=[sys.argv[0],'--config',str(resolved),'--output',str(out/'training')]
if args.resume:
    sys.argv+=['--resume',str(Path(args.resume).resolve())]
    if args.resume_sector is not None:sys.argv+=['--resume-sector',str(args.resume_sector)]
cli.main()
