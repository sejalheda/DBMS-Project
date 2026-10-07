const express = require("express");

const app = express();

app.use(express.json());

app.get("/api/health", (req, res) => {
    res.json({
        service: "CivicConnect Node Service",
        status: "running"
    });
});

app.get("/api/info", (req, res) => {
    res.json({
        project: "CivicConnect",
        purpose: "Public Grievance Tracking",
        framework: "Node.js + Express"
    });
});

const PORT = 5001;

app.listen(PORT, () => {
    console.log(`CivicConnect Node service running on port ${PORT}`);
});