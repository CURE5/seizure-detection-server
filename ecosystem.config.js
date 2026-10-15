// PM2 process definitions for the Raspberry Pi.
//
//   pm2 start ecosystem.config.js            # both processes
//   pm2 start ecosystem.config.js --only server
//
// Paths default to ~/seizure-detection-server and ~/seizure-detection-app.
// Override with SERVER_DIR / APP_DIR / VENV_DIR in the environment.
const path = require("path");

const HOME = process.env.HOME || "/home/pi";
const SERVER_DIR = process.env.SERVER_DIR || path.join(HOME, "seizure-detection-server");
const APP_DIR = process.env.APP_DIR || path.join(HOME, "seizure-detection-app");
const VENV_DIR = process.env.VENV_DIR || "sz";

module.exports = {
  apps: [
    {
      name: "app",
      cwd: APP_DIR,
      script: "yarn",
      args: "start",
      interpreter: "none",
      out_file: path.join(APP_DIR, "logs/app-out.log"),
      error_file: path.join(APP_DIR, "logs/app-err.log"),
      env: { NODE_ENV: "production" },
    },
    {
      name: "server",
      cwd: SERVER_DIR,
      script: path.join(SERVER_DIR, VENV_DIR, "bin/python"),
      args: "-m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level warning",
      interpreter: "none",
      out_file: path.join(SERVER_DIR, "logs/server-out.log"),
      error_file: path.join(SERVER_DIR, "logs/server-err.log"),
    },
  ],
};
