# pip install locust
# Run "locust -f locust_test.py"
from locust import HttpUser, task, between

class StressTest(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def test_text_endpoint(self):
        files = {'file': open('catdog.jpg', 'rb')}
        
        response = self.client.post("/predict?text=a photo of a cat, a photo of a dog, a photo of a lion", files=files)