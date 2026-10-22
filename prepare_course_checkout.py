"""Stage a source-only course folder in an explicitly supplied checkout.

This helper never commits, pushes or opens a PR. It replaces only this student's
project folder after archiving the previous folder outside the target checkout.
Historical study sources are copied without changing their locked contents.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def prepare(checkout, backup_name='course_folder_before_integration.tar.gz'):
    root=Path(__file__).resolve().parent
    source=root/'recognition/YOLO_s4905914'
    target=checkout/'recognition/YOLO_s4905914'
    if not (checkout/'.git').exists() or checkout.resolve()==root.resolve():
        raise ValueError('Supply the separate course checkout')
    audit=root/'outputs/closeout_20261022';audit.mkdir(parents=True,exist_ok=True)
    if Path(backup_name).name != backup_name or not backup_name.endswith('.tar.gz'):
        raise ValueError('Use a plain .tar.gz backup filename')
    backup=audit/backup_name
    if target.exists():
        if backup.exists():
            raise ValueError('Prior course-folder backup already exists; inspect before restaging')
        with tarfile.open(backup,'w:gz') as out:out.add(target,arcname='YOLO_s4905914')
        shutil.rmtree(target)
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','research_sources'))
    sources=target/'research_sources';sources.mkdir()
    members=[]
    # Respect Git exclusions: a staged final-evaluation bundle contains private
    # manifests and weights that must not enter the source-only course folder.
    study_files=subprocess.check_output(['git','-C',str(root),'ls-files','-z',
        '--cached','--others','--exclude-standard','--','studies'],text=True).split('\0')
    for relative in sorted(set(study_files)-{''}):
        file=root/relative
        if not file.is_file() or '__pycache__' in file.parts:continue
        if file.suffix in ('.pt','.pth','.npz','.zip','.gz','.pyc'):continue
        if file.name=='policy_predictions_seed3710.json':continue
        members.append(file)
    for name in ['PROTOCOL.json','MODEL_COMPONENTS_zh.txt','COURSE_REQUIREMENTS_AUDIT_zh.txt',
                 'CLOSEOUT_STATUS_zh.txt','TRANSFER_GUIDE_zh.txt','FEASIBILITY_REVIEW.txt']:
        members.append(root/name)
    hashes={}
    for file in members:
        relative=file.relative_to(root)
        dest=sources/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest)
        hashes[str(relative)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    manifest={'purpose':'Restore historical research paths without rewriting frozen experiment sources',
              'files':hashes,'photos_masks_training_labels_metadata_weights_included':False,
              'raw_prediction_cache_included':False,'source_workspace_git_commit':None}
    manifest['source_workspace_git_commit']=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    (sources/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'.gitignore').write_text('__pycache__/\n*.py[cod]\n*.pt\n*.pth\n.venv/\noutput/\noutputs/\ndata/\nweights/\n.DS_Store\n')
    verification={'source':str(source),'target':str(target),'research_source_files':len(hashes),
                  'module_files_identical':all((target/p.relative_to(source)).read_bytes()==p.read_bytes()
                    for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts),
                  'outside_project_folder_changed':False,'push_performed':False,'PR_created':False}
    assert verification['module_files_identical']
    (audit/'course_folder_preparation.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--backup-name',default='course_folder_before_integration.tar.gz')
    a=p.parse_args();prepare(a.checkout.resolve(),a.backup_name)
