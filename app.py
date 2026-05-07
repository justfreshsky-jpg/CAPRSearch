"""
CAPR Search — plain-language Q&A over Civil Air Patrol regulations and pamphlets.

CAP members ask: "what does CAPR 60-3 say about cadet supervision?" or
"how long is CPP good for?" — the app answers using its training-data
knowledge + a curated index of active CAPRs/CAPPs, and cites the
relevant regulation. Members verify against the current authoritative
version at capmembers.com.

Public-domain content (CAPRs/CAPPs are openly published on capmembers.com).
No PII. No member rosters. No eServices integration.

Built by a CAP member as a free volunteer offering.
"""
import collections
import functools
import json
import logging
import os
import re
import threading

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))
app.config.update(
    SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() == 'true',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('capr')


_metrics = {'requests_total': 0, 'provider_success': collections.Counter(), 'provider_failure': collections.Counter()}
_metrics_lock = threading.Lock()


def _route_handler(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            logger.exception('Unhandled exception in %s', f.__name__)
            return jsonify(error='An error occurred. Please try again.'), 500
    return wrapper


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    resp.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return resp


_HTTP_TIMEOUT = 35


def _llm_via_groq(system, user):
    key = os.environ.get('GROQ_KEY', '')
    if not key: return None
    r = requests.post('https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.3, 'response_format': {'type': 'json_object'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_cerebras(system, user):
    key = os.environ.get('CEREBRAS_KEY', '')
    if not key: return None
    r = requests.post('https://api.cerebras.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('CEREBRAS_MODEL', 'llama-3.3-70b'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.3},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_gemini(system, user):
    key = os.environ.get('GEMINI_KEY', '')
    if not key: return None
    r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}',
        headers={'Content-Type':'application/json'},
        json={'system_instruction':{'parts':[{'text':system}]},
              'contents':[{'role':'user','parts':[{'text':user}]}],
              'generationConfig':{'temperature':0.3, 'responseMimeType':'application/json'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['candidates'][0]['content']['parts'][0]['text']


def _llm_via_mistral(system, user):
    key = os.environ.get('MISTRAL_KEY', '')
    if not key: return None
    r = requests.post('https://api.mistral.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('MISTRAL_MODEL', 'mistral-small-latest'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.3, 'response_format': {'type': 'json_object'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_huggingface(system, user):
    key = os.environ.get('HF_KEY', '')
    if not key: return None
    r = requests.post('https://router.huggingface.co/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('HF_MODEL', 'meta-llama/Llama-3.3-70B-Instruct'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.3},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_sambanova(system, user):
    key = os.environ.get('SAMBANOVA_KEY', '')
    if not key: return None
    r = requests.post('https://api.sambanova.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('SAMBANOVA_MODEL', 'Meta-Llama-3.3-70B-Instruct'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}], 'temperature': 0.4},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_cloudflare(system, user):
    key = os.environ.get('CLOUDFLARE_AI_TOKEN', '')
    acct = os.environ.get('CLOUDFLARE_ACCOUNT_ID', '')
    if not key or not acct: return None
    model = os.environ.get('CLOUDFLARE_MODEL', '@cf/meta/llama-3.3-70b-instruct-fp8-fast')
    r = requests.post(f'https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}',
        headers={'Authorization': f'Bearer {key}'},
        json={'messages': [{'role':'system','content':system}, {'role':'user','content':user}], 'temperature': 0.4},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j.get('result', {}).get('response') or j.get('result', {}).get('output') or ''

_PROVIDERS = [
    ('groq', _llm_via_groq),
    ('cerebras', _llm_via_cerebras),
    ('gemini', _llm_via_gemini),
    ('mistral', _llm_via_mistral),
    ('huggingface', _llm_via_huggingface),
    ('sambanova', _llm_via_sambanova),
    ('cloudflare', _llm_via_cloudflare),
]


def _llm(system: str, user: str) -> str:
    last_err = None
    for name, fn in _PROVIDERS:
        try:
            out = fn(system, user)
            if out:
                with _metrics_lock:
                    _metrics['provider_success'][name] += 1
                return out.strip()
        except Exception as e:
            last_err = e
            with _metrics_lock:
                _metrics['provider_failure'][name] += 1
            logger.warning('Provider %s failed: %s', name, e)
    raise RuntimeError(f'All LLM providers failed: {last_err}')


# Curated index of active CAPRs and CAPPs. Reg numbers occasionally change
# during NHQ renumbering; the prompt directs users to verify the current
# version on capmembers.com. The LLM uses its broader CAP training-data
# knowledge to handle reg numbers not listed here.
CAPR_INDEX = {
    'CAPR 1-1': 'Civil Air Patrol Vision, Mission, Core Values, and History',
    'CAPR 7-2': 'IT Acceptable Use, Data Stewardship, and Records',
    'CAPR 11-3': 'Reporting Requirements (incidents, recurring submissions)',
    'CAPR 20-1': 'Organization of Civil Air Patrol',
    'CAPR 30-1': 'Senior Member Membership (eligibility, application, dues)',
    'CAPR 30-2': 'Cadet Membership (ages 12 to under 21)',
    'CAPR 31-1': 'Personnel Promotions (senior member grades)',
    'CAPR 35-3': 'Membership Termination and Disciplinary Action',
    'CAPR 36-1': 'Equal Opportunity and Anti-Discrimination',
    'CAPR 39-1': 'CAP Uniform Manual (how to wear every uniform)',
    'CAPR 39-2': 'CAP Awards (medals and ribbons authority)',
    'CAPR 39-3': 'Award of CAP Medals, Ribbons, and Certificates (criteria, nomination, approval)',
    'CAPR 50-15': 'Special Activities (national and regional cadet activities)',
    'CAPR 50-17': 'Senior Member Professional Development Program (Levels 1 through 5)',
    'CAPR 52-10': 'Cadet Protection Program (CPP) — child safety, training requirements (renews every 2 years)',
    'CAPR 52-16': 'Cadet Program Management (achievements, ranks, promotions, attendance)',
    'CAPR 60-1': 'Aviation Standardization and Evaluation (Form 5 checks)',
    'CAPR 60-3': 'Emergency Services Training and Operational Missions (qualifications: GES, MS, MO, MP, MSA, MSO, IC, etc.)',
    'CAPR 60-4': 'Search and Rescue and Disaster Relief Procedures',
    'CAPR 62-1': 'Safety Program (squadron-level program management)',
    'CAPR 62-2': 'Mishap Reporting, Review, and CAPF 78',
    'CAPR 66-1': 'Aircraft Maintenance Management',
    'CAPR 70-1': 'Aerospace Education Program',
    'CAPR 77-1': 'Operation and Maintenance of CAP Vehicles',
    'CAPR 110-1': 'Communications (radio operators, COMSEC, equipment)',
    'CAPR 173-1': 'Financial Procedures (squadron and wing fiscal management)',
    'CAPR 174-1': 'Property and Logistics Management',
    'CAPP 1': 'Core Values pamphlet (training resource for the four core values)',
    'CAPP 50-2': 'Leadership 2000 (senior member professional development guide)',
    'CAPP 50-9': 'Senior Member Curriculum task lookup',
    'CAPP 51-1': 'Cadet Leadership Lab activity guide',
    'CAPP 60-50': 'Cadet Programs Management practical handbook',
    'CAPP 60-31': 'Cadet Achievement curriculum (per-achievement study guide)',
    'CAPP 70-1': 'Aerospace Education Excellence (AEX) program guide',
}


def _format_index() -> str:
    return '\n'.join(f'  {k}: {v}' for k, v in CAPR_INDEX.items())


_CAPR_SYSTEM = (
    "You are a Civil Air Patrol regulation Q&A assistant. CAP members ask plain-language questions "
    "about CAP regulations (CAPRs) and pamphlets (CAPPs); you answer using your training-data knowledge "
    "of CAP regulations, citing the specific reg + section number. Always include a verify-current-version reminder.\n\n"
    "Output a JSON object with these fields:\n"
    '{\n'
    '  "answer": "concise plain-language answer, 2-6 sentences",\n'
    '  "primary_reg": "the main regulation cited (e.g., \'CAPR 60-3\')",\n'
    '  "section_or_paragraph": "section number, paragraph, table, or attachment reference if known (e.g., \'§4.2\' or \'Table 4-1\') — null if uncertain",\n'
    '  "secondary_regs": ["array of other CAPRs/CAPPs that also touch this topic"],\n'
    '  "key_caveats": ["array of important caveats, exceptions, or related compliance notes"],\n'
    '  "confidence": "high | medium | low — your confidence in this answer being current and accurate",\n'
    '  "verify_url": "https://www.capmembers.com/forms_publications__regulations/"\n'
    '}\n\n'
    "RULES:\n"
    "- Output ONLY the JSON object. No prose around it.\n"
    "- If the question is outside CAP scope (general aviation, USAF active-duty issues unrelated to CAP, personal legal advice), set confidence to 'low' and explain in 'answer' that the user should consult the appropriate authority.\n"
    "- If you don't know which reg covers something, set primary_reg to null and confidence to 'low'. Do NOT invent reg numbers.\n"
    "- CAP renumbers regulations periodically. Always include verify_url and recommend the user verify the current version at capmembers.com.\n"
    "- For sensitive topics (Cadet Protection allegations, financial irregularities, mishap reporting): answer the procedural question but flag in key_caveats that the user should engage the proper chain of command (squadron CC -> Wing IG/JA/Safety) immediately.\n"
    "- Never give legal, medical, or aviation safety-of-flight advice. Cite the reg and recommend the appropriate authority.\n\n"
    "ACTIVE CAP REGULATIONS AND PAMPHLETS (curated index — use general training knowledge for others):\n"
    + _format_index()
)


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith('```'):
        s = re.sub(r'^```[a-zA-Z]*\s*', '', s)
        s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify(status='ok')


@app.route('/metrics')
def metrics():
    with _metrics_lock:
        return jsonify({
            'requests_total': _metrics['requests_total'],
            'provider_success': dict(_metrics['provider_success']),
            'provider_failure': dict(_metrics['provider_failure']),
        })


@app.route('/api/ask', methods=['POST'])
@_route_handler
def ask():
    data = request.get_json(silent=True) or {}
    q = (data.get('question') or '').strip()
    if not q:
        return jsonify(error='Please enter a question.'), 400
    if len(q) > 1500:
        return jsonify(error='Question is too long (max 1500 characters).'), 400
    with _metrics_lock:
        _metrics['requests_total'] += 1
    raw = _llm(_CAPR_SYSTEM, f'CAP MEMBER QUESTION:\n\n{q}')
    raw = _strip_code_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning('LLM returned non-JSON: %s', raw[:200])
        return jsonify(error='The model returned an unparseable response. Please try again.'), 502
    return jsonify(result=parsed)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
