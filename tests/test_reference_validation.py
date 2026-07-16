import json
import sys
import types
import unittest
from unittest.mock import patch


class _StubLLMChain:
    def __init__(self, **_kwargs):
        pass

    def complete(self, **_kwargs):
        return None


try:
    import freshsky_common.llm  # noqa: F401
except ModuleNotFoundError:
    freshsky_common = types.ModuleType('freshsky_common')
    freshsky_common_llm = types.ModuleType('freshsky_common.llm')
    freshsky_common_llm.LLMChain = _StubLLMChain
    freshsky_common_llm.install_provider_metrics = lambda _app: None
    sys.modules.setdefault('freshsky_common', freshsky_common)
    sys.modules.setdefault('freshsky_common.llm', freshsky_common_llm)

import app as capr_app  # noqa: E402


class ReferenceValidationTests(unittest.TestCase):
    def test_current_index_replaces_materially_wrong_mappings(self):
        index = capr_app.CAPR_INDEX
        self.assertEqual(index['CAPR 20-1']['title'], 'Inspector General Program')
        self.assertEqual(index['CAPR 30-1']['title'], 'Organization of Civil Air Patrol')
        self.assertEqual(index['CAPR 39-2']['title'], 'Civil Air Patrol Membership')
        self.assertEqual(index['CAPR 60-1']['title'], 'Cadet Program Management')
        self.assertEqual(index['CAPR 60-2']['title'], 'Cadet Protection Program')
        self.assertEqual(index['CAPR 60-3']['title'], 'Cadets At School Program')
        self.assertEqual(
            index['CAPR 60-3(I)']['title'],
            'CAP Emergency Services Training and Operational Missions',
        )
        self.assertEqual(index['CAPR 70-1']['title'], 'CAP Flight Management')
        self.assertEqual(index['CAPR 110-1']['title'], 'Civil Air Patrol History Program')

        for stale in (
            'CAPR 7-2', 'CAPR 11-3', 'CAPR 30-2', 'CAPR 31-1',
            'CAPR 50-15', 'CAPR 50-17', 'CAPR 52-10', 'CAPR 52-16',
            'CAPR 60-4', 'CAPR 62-1', 'CAPR 62-2', 'CAPR 66-1', 'CAPR 77-1',
        ):
            self.assertNotIn(stale, index)

    def test_supported_citation_gets_server_owned_metadata(self):
        self.assertEqual(
            capr_app._validated_publication('CAPR 60-3 (I)'),
            'CAPR 60-3(I)',
        )
        result = capr_app._normalize_result({
            'answer': 'Cadet Protection is governed by the current cadet-protection regulation.',
            'primary_reg': 'CAPR 60-2',
            'section_or_paragraph': '§4.2',
            'secondary_regs': ['CAPR 60-1', 'CAPR 52-10', 'MADE UP 1'],
            'key_caveats': [],
            'confidence': 'high',
            'verify_url': 'https://untrusted.example/',
        })

        self.assertEqual(result['primary_reg'], 'CAPR 60-2')
        self.assertIsNone(result['section_or_paragraph'])
        self.assertEqual(result['secondary_regs'], ['CAPR 60-1'])
        self.assertEqual(result['confidence'], 'medium')
        self.assertEqual(result['citation_status'], 'index-validated')
        self.assertEqual(result['source_title'], 'Cadet Protection Program')
        self.assertEqual(result['source_version'], '18 Aug 2025')
        self.assertEqual(result['source_retrieved'], '2026-07-16')
        self.assertEqual(result['verify_url'], capr_app.CAP_REGULATIONS_URL)

    def test_unsupported_citation_is_refused(self):
        result = capr_app._normalize_result({
            'answer': 'A confident but unsupported model answer.',
            'primary_reg': 'CAPR 52-10',
            'section_or_paragraph': '2.3',
            'secondary_regs': [],
            'key_caveats': [],
            'confidence': 'high',
        })

        self.assertIsNone(result['primary_reg'])
        self.assertIsNone(result['section_or_paragraph'])
        self.assertEqual(result['confidence'], 'low')
        self.assertEqual(result['citation_status'], 'unsupported')
        self.assertIn("can't support a current CAP publication citation", result['answer'])

    def test_api_applies_validation_after_model_output(self):
        model_payload = json.dumps({
            'answer': 'Use the old number.',
            'primary_reg': 'CAPR 52-10',
            'section_or_paragraph': '3.1',
            'secondary_regs': [],
            'key_caveats': [],
            'confidence': 'high',
        })
        with patch.object(capr_app, '_llm', return_value=model_payload):
            response = capr_app.app.test_client().post(
                '/api/ask', json={'question': 'Which current publication covers CPP?'}
            )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()['result']
        self.assertIsNone(result['primary_reg'])
        self.assertEqual(result['citation_status'], 'unsupported')


if __name__ == '__main__':
    unittest.main()
