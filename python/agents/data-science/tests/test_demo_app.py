import unittest
import json
from demo.app import app

class TestDemoApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        """Test that the index page loads."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ADK Data Science Agent Demo', response.data)

    # Note: We won't run a full /chat test here as it requires 
    # mock models or real API keys/credentials which might not be 
    # set up in the test environment. 
    # However, we can check that it handles missing data.
    def test_chat_no_message(self):
        """Test chat endpoint with missing message."""
        response = self.app.post('/chat', 
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No message provided')

if __name__ == '__main__':
    unittest.main()
