from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class APIHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/product":

            response = {
                "name": "MacBook Air",
                "available": True
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            self.wfile.write(
                json.dumps(response).encode()
            )

        else:

            self.send_response(404)
            self.end_headers()


server = HTTPServer(
    ("0.0.0.0", 8000),
    APIHandler
)

print("API running on port 8000")

server.serve_forever()