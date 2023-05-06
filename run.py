import uvicorn
if __name__ == "__main__":
    config = uvicorn.Config("routers_1:app", host='0.0.0.0', port=5000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
