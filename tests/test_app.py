import unittest
from app import app

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        # Set up test client
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<form', response.data)
        self.assertIn(b'name="hours_studied"', response.data)

    def test_predict_valid_input(self):
        data = {
            'hours_studied': '10',
            'attendance_rate': '85',
            'previous_scores': '75'
        }
        response = self.app.post('/predict', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted academic performance:', response.data)

    def test_predict_missing_input(self):
        data = {
            'hours_studied': '10',
            'attendance_rate': '',
            'previous_scores': '75'
        }
        response = self.app.post('/predict', data=data)
        self.assertIn(b'Missing input for', response.data)
        self.assertEqual(response.status_code, 400)

    def test_predict_invalid_input(self):
        data = {
            'hours_studied': 'ten',
            'attendance_rate': '85',
            'previous_scores': '75'
        }
        response = self.app.post('/predict', data=data)
        self.assertIn(b'Invalid input for', response.data)
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
