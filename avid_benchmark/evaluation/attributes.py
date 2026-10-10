"""Score frozen qwen_v3 answers independently of expensive image inference."""
from collections import defaultdict
from .common import MetricResult, result


def _rows(rows, attribute):
    selected=[r for r in rows if r['attribute']==attribute and r.get('pair_type','factual')=='factual']
    if not selected:
        raise ValueError(f'No factual answers for {attribute}.')
    if any(r.get('bank_version','qwen_v3')!='qwen_v3' for r in selected):
        raise ValueError('These scores require the frozen qwen_v3 question bank.')
    if any('.qwen_v' in str(r.get('question_id','')) and not str(r['question_id']).endswith('.qwen_v3') for r in selected):
        raise ValueError('Question IDs identify a different Qwen bank version.')
    return selected


def _score(rows, attribute, balanced=False, chance=False):
    rows=_rows(rows,attribute)
    values=[]
    groups=defaultdict(list)
    for row in rows:
        choices=row['choices']
        accepted=set(row.get('acceptable_answers') or [row['answer']])
        if not choices or not accepted.issubset(set(choices)):
            raise ValueError('Accepted labels must belong to nonempty choices.')
        value=len(accepted)/len(choices) if chance else float(row.get('mapped_answer','') in accepted)
        values.append(value)
        groups[str(row['answer'])].append(value)
    name=f'{attribute}_' + ('balanced_' if balanced else '') + ('chance' if chance else 'accuracy')
    raw=result(name,values,[f"{r['image_name']}:{r.get('question_id',attribute)}" for r in rows],'avid-qwen-v3')
    if not balanced:
        return raw
    recalls={label:sum(v)/len(v) for label,v in groups.items()}
    return MetricResult(name,sum(recalls.values())/len(recalls),raw.sample_ids,raw.values,raw.protocol,{'class_scores':recalls,'class_counts':{k:len(v) for k,v in groups.items()}})


def compute_color_accuracy(rows):
    return _score(rows,'color')

def compute_position_accuracy(rows):
    return _score(rows,'position')

def compute_identity_accuracy(rows):
    return _score(rows,'visual_identity')

def compute_color_balanced_accuracy(rows):
    return _score(rows,'color',balanced=True)

def compute_position_balanced_accuracy(rows):
    return _score(rows,'position',balanced=True)

def compute_identity_balanced_accuracy(rows):
    return _score(rows,'visual_identity',balanced=True)

def compute_color_balanced_chance(rows):
    return _score(rows,'color',balanced=True,chance=True)

def compute_position_balanced_chance(rows):
    return _score(rows,'position',balanced=True,chance=True)

def compute_identity_balanced_chance(rows):
    return _score(rows,'visual_identity',balanced=True,chance=True)


def extract_attributes(images, question_bank, evaluator):
    """images is an ID→PIL map; evaluator exposes answer(image,prompt,choices)."""
    if question_bank.get('version',question_bank.get('bank_version')) != 'qwen_v3':
        raise ValueError('Supply the frozen qwen_v3 bank.')
    rows=[]
    for entry in question_bank['entries']:
        name=entry['image_name']
        if name not in images:
            continue
        for qa in entry['qas']:
            prompt=qa.get('prompt',qa.get('question',''))
            free,mapped=evaluator.answer(images[name],prompt,qa['choices'])
            rows.append({**qa,'image_name':name,'bank_version':'qwen_v3','free_form_answer':free,'mapped_answer':mapped})
    if {r['image_name'] for r in rows} != set(images):
        raise ValueError('Question bank does not cover every requested image.')
    return rows
