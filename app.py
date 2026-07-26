"""Plain-language Q&A over a dated CAP NHQ publication-index snapshot.

The model may summarize, but server-side validation permits only publication
numbers present in the verified index below. Paragraph-level citations are
withheld because the app does not retrieve authoritative full-text documents.
No PII, member rosters, or eServices integration.
"""
from __future__ import annotations

import collections
import functools
import json
import logging
import os
import re
import threading

from flask import Response, Flask, jsonify, render_template, request

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))
app.config.update(
    SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() == 'true',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

from freshsky_common.revenue import install_visuals  # noqa: E402
install_visuals(app)
from freshsky_common.freemium import register_freemium  # noqa: E402
from freshsky_common.hulec import install_hulec  # noqa: E402
from freshsky_common.security import install_security_headers  # noqa: E402

register_freemium(
    app,
    primary_url=os.environ.get('APP_URL', 'https://capr.freshskyai.com'),
    community_mode=True,
    gate_all_post=True,
    workspace_id='civic',
)
install_hulec(app, slug='capr')
install_security_headers(app)

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


# Provider calls are centralized in the privacy-restricted shared chain.

from freshsky_common.llm import LLMChain, install_provider_metrics  # noqa: E402

_SHARED_LLM = LLMChain(privacy_profile="us_public")
install_provider_metrics(app)


def _llm_via_shared_chain(system, user):
    return _SHARED_LLM.complete(system=system, user=user) or None


_PROVIDERS = [('shared', _llm_via_shared_chain)]


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


SOURCE_RETRIEVED = '2026-07-16'
CAP_REGULATIONS_URL = (
    'https://www.gocivilairpatrol.com/members/publications/'
    'indexes-regulations-and-manuals-1700'
)
CAP_PAMPHLETS_URL = (
    'https://www.gocivilairpatrol.com/members/publications/pamphlets-1702/'
)


def _publication(title: str, version: str, source_url: str = CAP_REGULATIONS_URL) -> dict:
    return {
        'title': title,
        'version': version,
        'source_url': source_url,
        'retrieved': SOURCE_RETRIEVED,
    }


# Exact metadata transcribed from the official CAP NHQ public indexes on the
# retrieval date above. This is intentionally finite: unlisted citations are
# rejected server-side instead of being guessed from model training data.
CAPR_INDEX = {
    'CAPR 1-1': _publication('Ethics Policy', '15 Mar 2012'),
    'CAPR 1-2': _publication('Publications Management', '7 Nov 2016'),
    'CAPR 1-2(I)': _publication('Personally Identifiable Information', '3 Apr 2012'),
    'CAPR 20-1': _publication('Inspector General Program', '30 Sep 2025'),
    'CAPR 20-2': _publication('Complaint Resolution', '7 Nov 2025'),
    'CAPR 20-3': _publication('Inspections and Compliance Analyses Implementation Guide', '18 May 2026'),
    'CAPR 30-1': _publication('Organization of Civil Air Patrol', '13 Jan 2020; includes current ICLs'),
    'CAPR 35-1': _publication('Assignment and Duty Status', '4 Jun 2015'),
    'CAPR 35-3': _publication('Membership Termination', '27 Dec 2012'),
    'CAPR 35-5': _publication('CAP Officer and NCO Appointments and Promotions', '22 Nov 2016'),
    'CAPR 36-1': _publication('Civil Air Patrol Nondiscrimination Program', '14 May 2026'),
    'CAPR 36-2': _publication('Complaints Under the Civil Air Patrol Nondiscrimination Policy', '14 May 2026'),
    'CAPR 39-1': _publication('Civil Air Patrol Uniform Regulation', '3 Mar 2020; includes current ICLs'),
    'CAPR 39-2': _publication('Civil Air Patrol Membership', '18 Aug 2025'),
    'CAPR 39-3': _publication('Award of CAP Medals, Ribbons and Certificates', '28 Dec 2012'),
    'CAPR 39-4': _publication('Operations Ratings, Awards and Badges', '21 Apr 2025'),
    'CAPR 40-1': _publication('Civil Air Patrol Senior Member Education & Training Program', '24 May 2021'),
    'CAPR 40-2': _publication('Testing Administration and Security', '1 Jan 2018'),
    'CAPR 50-1': _publication('Aerospace Education Mission', '9 Nov 2020'),
    'CAPR 60-1': _publication('Cadet Program Management', '18 Aug 2025'),
    'CAPR 60-2': _publication('Cadet Protection Program', '18 Aug 2025'),
    'CAPR 60-3': _publication('Cadets At School Program', '26 Oct 2021'),
    'CAPR 60-3(I)': _publication('CAP Emergency Services Training and Operational Missions', '26 Dec 2012'),
    'CAPR 60-5': _publication('Critical Incident Stress Management', '3 Nov 2006'),
    'CAPR 60-6': _publication('CAP Counterdrug Operations', '26 Dec 2012'),
    'CAPR 70-1': _publication('CAP Flight Management', '31 Mar 2020'),
    'CAPR 70-4': _publication('CAP sUAS Flight Management', '9 Jan 2023'),
    'CAPR 100-1': _publication('Radio Communications Management', '6 Apr 2016'),
    'CAPR 100-3': _publication('Radiotelephone Operations', '6 Apr 2016'),
    'CAPR 103-1': _publication('Payment for Mission Support', '17 Oct 2025'),
    'CAPR 110-1': _publication('Civil Air Patrol History Program', '25 Jan 2021'),
    'CAPR 120-1': _publication('Information Technology Security', '21 Sep 2022'),
    'CAPR 130-2': _publication('Civil Air Patrol Aircraft Maintenance Management', '4 Oct 2021; includes current ICLs'),
    'CAPR 132-1': _publication('CAP Vehicle Management', '29 May 2025; includes current ICLs'),
    'CAPR 160-1': _publication('Civil Air Patrol Safety Program', '1 Nov 2019'),
    'CAPR 160-2': _publication('Safety Reporting and Review', '28 Dec 2022'),
    'CAPR 173-1': _publication('Financial Procedures and Accounting', '15 Nov 2012; includes current ICLs'),
    'CAPR 174-1': _publication('Property Management and Accountability', '26 Dec 2012; includes current ICLs'),
    'CAPR 190-1': _publication('Civil Air Patrol Public Affairs Program', '16 Nov 2016; includes current ICLs'),
    'CAPP 40-35': _publication('Command Specialty Track Study Guide', '29 Jun 2026', CAP_PAMPHLETS_URL),
    'CAPP 60-11': _publication('Cadet Programs Officer Handbook and Specialty Track Guide', 'Jun 2026', CAP_PAMPHLETS_URL),
    'CAPP 70-1': _publication('Operations Officer Specialty Track Study Guide', '17 Oct 2025; supersedes CAPP 211', CAP_PAMPHLETS_URL),
    'CAPP 70-3': _publication('Emergency Services Officer Specialty Track Study Guide', '17 Oct 2025', CAP_PAMPHLETS_URL),
    'CAPP 200': _publication('Personnel Officer Specialty Track Study Guide', '30 Apr 2026', CAP_PAMPHLETS_URL),
}


def _format_index() -> str:
    return '\n'.join(
        f"  {number}: {entry['title']} (version/date: {entry['version']})"
        for number, entry in CAPR_INDEX.items()
    )


_CAPR_SYSTEM = (
    "You are a Civil Air Patrol publication-index Q&A assistant. The authoritative material supplied "
    "to you is INDEX METADATA ONLY: publication number, title, and version/date. You do not have the "
    "current full text. Use the index to identify which current publication likely governs a topic, but "
    "do not claim that training-data memory is current authority.\n\n"
    "Output a JSON object with these fields:\n"
    '{\n'
    '  "answer": "concise plain-language answer, 2-6 sentences",\n'
    '  "primary_reg": "the main publication cited (e.g., \'CAPR 60-2\')",\n'
    '  "section_or_paragraph": null,\n'
    '  "secondary_regs": ["array of other CAPRs/CAPPs that also touch this topic"],\n'
    '  "key_caveats": ["array of important caveats, exceptions, or related compliance notes"],\n'
    '  "confidence": "high | medium | low — your confidence in this answer being current and accurate",\n'
    f'  "verify_url": "{CAP_REGULATIONS_URL}"\n'
    '}\n\n'
    "RULES:\n"
    "- Output ONLY the JSON object. No prose around it.\n"
    "- If the question is outside CAP scope (general aviation, USAF active-duty issues unrelated to CAP, personal legal advice), set confidence to 'low' and explain in 'answer' that the user should consult the appropriate authority.\n"
    "- primary_reg and secondary_regs may contain ONLY exact publication numbers from the supplied index.\n"
    "- Always set section_or_paragraph to null. The index does not support paragraph-level citations.\n"
    "- If the question asks for a ratio, interval, permission, exception, procedure, exact requirement, "
    "or other substantive rule that cannot be verified from the index metadata alone, say so plainly, "
    "give only a source-finding lead, and use low confidence.\n"
    "- If you do not know which indexed publication covers something, set primary_reg to null and "
    "confidence to low. Never invent or modernize a number from memory.\n"
    "- Always include the supplied verify_url and tell the user to open the current official document.\n"
    "- For sensitive topics (Cadet Protection allegations, financial irregularities, mishap reporting): do not invent procedural steps from memory. Direct the user to the current publication and the responsible command/IG/JA/Safety channel.\n"
    "- Never give legal, medical, or aviation safety-of-flight advice. Cite the reg and recommend the appropriate authority.\n\n"
    "CURRENT CAP PUBLICATION INDEX SNAPSHOT (official CAP NHQ indexes, retrieved 2026-07-16):\n"
    + _format_index()
)


def _validated_publication(value) -> str | None:
    """Return an exact indexed publication number or refuse the citation."""
    if not isinstance(value, str):
        return None
    candidate = re.sub(r'\s+', ' ', value.strip()).upper()
    candidate = re.sub(r'\s+\(I\)$', '(I)', candidate)
    for number in CAPR_INDEX:
        if candidate == number.upper():
            return number
    return None


def _normalize_result(parsed) -> dict:
    """Attach trusted metadata and remove unsupported model citations."""
    if not isinstance(parsed, dict):
        parsed = {}

    answer = parsed.get('answer')
    if not isinstance(answer, str) or not answer.strip():
        answer = 'No supported answer was returned.'
    else:
        answer = answer.strip()

    primary = _validated_publication(parsed.get('primary_reg'))
    raw_secondary = parsed.get('secondary_regs')
    secondary = []
    if isinstance(raw_secondary, list):
        for item in raw_secondary:
            number = _validated_publication(item)
            if number and number != primary and number not in secondary:
                secondary.append(number)

    raw_caveats = parsed.get('key_caveats')
    caveats = [
        item.strip() for item in raw_caveats
        if isinstance(item, str) and item.strip()
    ][:6] if isinstance(raw_caveats, list) else []

    result = {
        'answer': answer,
        'primary_reg': primary,
        # Full current publication text is not loaded, so model-supplied paragraph
        # references are never presented as verified citations.
        'section_or_paragraph': None,
        'secondary_regs': secondary,
        'key_caveats': caveats,
        'confidence': 'low',
        'citation_status': 'unsupported',
        'source_title': None,
        'source_version': None,
        'source_url': CAP_REGULATIONS_URL,
        'source_retrieved': SOURCE_RETRIEVED,
        'verify_url': CAP_REGULATIONS_URL,
    }

    if primary is None:
        result['secondary_regs'] = []
        result['key_caveats'] = []
        result['answer'] = (
            "I can't support a current CAP publication citation for this answer from the "
            "verified index snapshot. Open the official CAP NHQ publications library or "
            "consult the responsible CAP office; do not rely on an uncited model answer."
        )
        result['key_caveats'].append(
            'The model did not return an exact publication number present in the verified index.'
        )
        return result

    source = CAPR_INDEX[primary]
    result.update({
        'source_title': source['title'],
        'source_version': source['version'],
        'source_url': source['source_url'],
        'source_retrieved': source['retrieved'],
        'citation_status': 'index-validated',
        'verify_url': source['source_url'],
        # Index validation confirms the publication identity, not the model's
        # substantive summary. High confidence would imply full-text checking.
        'confidence': 'medium' if parsed.get('confidence') in {'high', 'medium'} else 'low',
    })
    result['key_caveats'].append(
        'Publication number, title, and version were validated against the CAP NHQ index; '
        'the answer and any detailed rule were not checked against the full current document.'
    )
    return result


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
    return jsonify(result=_normalize_result(parsed))


_PRIVACY_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Privacy — CAPR Search</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;line-height:1.6;color:#0f172a}h1{margin-bottom:.5em}h2{margin-top:1.5em;font-size:1.1rem}a{color:#1e3a8a}</style>
</head><body>
<a href="/">← Back to CAPR Search</a>
<h1>Privacy Policy — CAPR Search</h1>
<p><em>Last updated 2026-07-26</em></p>
<h2>What we collect</h2>
<p>CAPR Search is a stateless tool. We do <strong>not</strong> store the text or voice input you submit. An email address is used only when you choose to sign in for paid access. CAPR Search does not accept or need member rosters, CAPIDs, patient health information (PHI), incident identifiers, or operational secrets.</p>
<h2>What we send to AI providers</h2>
<p>The text or voice transcript you submit is sent through the configured restricted U.S. AI provider pool. Provider availability can change. The shared privacy layer rejects several common identifier patterns before provider calls, but automated screening is not a substitute for removing identifying information. <strong>Do not submit rosters, CAPIDs, PHI, incident identifiers, operational secrets, payment data, or other sensitive data.</strong></p>
<h2>What gets logged</h2>
<p>Standard request metadata (IP address, timestamp, response code) is logged by Google Cloud Run for operational purposes (debugging, abuse prevention) and rotated automatically per Google retention defaults. We do not associate logs with individual users.</p>
<h2>Cookies</h2>
<p>A Flask session cookie is set to remember ephemeral state during your visit. It expires when you close the browser. No third-party tracking, no advertising cookies.</p>
<h2>Children</h2>
<p>Some of our tools (e.g. CAPStudy) are designed to be used by minors aged 12+. We do not collect any personally identifying information from anyone, including minors. Parents/guardians of cadets aged 12-17 may use the tool freely.</p>
<h2>Contact</h2>
<p>Questions: <a href="https://www.freshskyai.com/contact">Fresh Sky contact page</a>. Operator: Fresh Sky LLC, Somerset County, NJ.</p>
</body></html>"""

_TERMS_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Terms of Use — CAPR Search</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;line-height:1.6;color:#0f172a}h1{margin-bottom:.5em}h2{margin-top:1.5em;font-size:1.1rem}a{color:#1e3a8a}</style>
</head><body>
<a href="/">← Back to CAPR Search</a>
<h1>Terms of Use — CAPR Search</h1>
<p><em>Last updated 2026-07-26</em></p>
<h2>What this is</h2>
<p>CAPR Search is an unofficial, experimental, privacy-first Civic module offered by Fresh Sky LLC. Three previews are included; Civic access is $14.99/month with an allowance of 40 usage units per day and 200 per month, and may be canceled monthly. Civic access covers CivicOps and CAP modules only; it does not grant access to non-Civic Fresh Sky products. Existing eligible subscriptions continue to be recognized.</p>
<h2>What this is not</h2>
<p>CAPR Search is <strong>not</strong> an official Civil Air Patrol publication search system and is not affiliated with or endorsed by Civil Air Patrol, any government agency, or any military service. Output is AI-generated and intended as a source lead only — the human user is responsible for opening and verifying the authoritative current publication before acting.</p>
<h2>Use at your own discretion</h2>
<p>You agree to use the tool in good faith. <strong>Do not submit rosters, CAPIDs, personally identifying information (PII), patient health information (PHI), incident identifiers, operational secrets, or classified/sensitive operational details.</strong> The tool is not designed to handle such data and we do not warrant against any misuse.</p>
<h2>No warranty</h2>
<p>The tool is provided "as is" without warranty of any kind. Fresh Sky LLC disclaims all liability for damages arising from use or misuse of the output.</p>
<h2>Changes</h2>
<p>We may update or discontinue the tool without notice. If a tool is retired, this URL will redirect or be retired in tandem.</p>
<h2>Contact</h2>
<p>Questions: <a href="https://www.freshskyai.com/contact">Fresh Sky contact page</a>.</p>
</body></html>"""


@app.route('/robots.txt')
def _robots():
    return Response(
        "User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /metrics\nDisallow: /health\n"
        "Sitemap: https://capr.freshskyai.com/sitemap.xml\n",
        mimetype='text/plain',
    )


@app.route('/sitemap.xml')
def _sitemap():
    return Response(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        '  <url><loc>https://capr.freshskyai.com/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n'
        '</urlset>\n',
        mimetype='application/xml',
    )


@app.route('/privacy')
def _privacy():
    return Response(_PRIVACY_HTML, mimetype='text/html')


@app.route('/terms')
def _terms():
    return Response(_TERMS_HTML, mimetype='text/html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
