"""Check that API inputs reach the correct model and invalid inputs are rejected."""
import unittest
from fastapi.testclient import TestClient
from main import app, regression, classification


class AppTests(unittest.TestCase):
    def test_routes(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/metrics').status_code, 200)
            result = client.post('/predict/regression', json={'total_bill': 20})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['prediction'], round(float(regression.predict([[20]])[0]), 2))
            for length, width in [(1.4, 0.2), (4.5, 1.3), (5.8, 2.2)]:
                result = client.post('/predict/classification', json={'petal_length': length, 'petal_width': width})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['prediction'], classification.predict([[length, width]])[0])
            for bad in [{}, {'total_bill': -1}, {'total_bill': 100}, {'total_bill': 'abc'}, {'total_bill': 20, 'other': 1}]:
                self.assertEqual(client.post('/predict/regression', json=bad).status_code, 422)
            self.assertEqual(client.post('/predict/classification', json={'petal_length': 1.4}).status_code, 422)
            self.assertEqual(client.post('/predict/classification', json={'petal_length': 1.4, 'petal_width': 9}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
