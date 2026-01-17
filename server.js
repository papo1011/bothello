const express = require('express');
const path = require('path');
const {execFile, spawn} = require('child_process');
const readline = require('readline');

const app = express();
app.use(express.json());

const staticDir = path.join(__dirname, 'full_stack-othello_game');
app.use(express.static(staticDir));

const availableBots = ['cpu', 'omp', 'leaf', 'block', 'cuda'];

app.get('/api/bots', (req, res) => {
  res.json({bots: availableBots});
});

// Manage persistent bot sessions so a bot process can remain loaded in memory
// for the lifetime of a (client) session and avoid re-initialization overhead.
const sessions = new Map();
const SESSION_IDLE_MS = 1000 * 60 * 15; // 15 minutes

function createSession(bot, time_ms) {
  const cliPath = path.join(__dirname, 'build', 'bot_cli');
  const proc = spawn(cliPath, ['--bot', bot, '--server', '--time-ms', String(time_ms)]);

  const rl = readline.createInterface({ input: proc.stdout });
  const session = {
    proc,
    rl,
    pending: [],
    lastActive: Date.now(),
    timeout: null
  };

  // When a line is received from the bot, resolve the pending promise
  rl.on('line', (line) => {
    let parsed = null;
    try {
      parsed = JSON.parse(line);
    } catch (e) {
      console.error('Invalid bot output (not JSON):', line);
      return;
    }
    session.lastActive = Date.now();
    if (session.pending.length > 0) {
      const cb = session.pending.shift();
      cb(null, parsed);
    } else {
      // No pending request: this can happen on unexpected output
      console.warn('Bot produced unsolicited output:', parsed);
    }
  });

  proc.stderr.on('data', (d) => console.error('bot stderr:', d.toString()));
  proc.on('exit', (code, sig) => {
    console.log('Bot process exited', code, sig);
    rl.close();
  });

  // Idle timeout cleanup
  function refreshTimeout() {
    if (session.timeout) clearTimeout(session.timeout);
    session.timeout = setTimeout(() => {
      try {
        session.proc.kill();
      } catch (e) {}
      sessions.forEach((v, k) => { if (v === session) sessions.delete(k); });
    }, SESSION_IDLE_MS);
  }
  refreshTimeout();
  session.refreshTimeout = refreshTimeout;

  return session;
}

app.post('/api/bot/move', (req, res) => {
  const {bot, black, white, time_ms, is_black_turn, session: sessionId} = req.body;

  if (!availableBots.includes(bot)) {
    return res.status(400).json({error: 'Unknown bot'});
  }

  // If session is provided and exists, use persistent bot. Otherwise create a
  // new persistent session and return its id. If session is omitted, fall back
  // to single-shot mode (backwards compatible).
  if (sessionId) {
    const sess = sessions.get(sessionId);
    if (!sess) {
      return res.status(400).json({error: 'Unknown session'});
    }

    return sendToSession(sess, black, white, time_ms, is_black_turn, (err, parsed) => {
      if (err) return res.status(500).json({error: 'bot error', details: err.toString()});
      res.json(Object.assign({}, parsed, {session: sessionId}));
    });
  }

  // No session requested: create a new session for this client and return its id
  // so the client may reuse it for subsequent moves.
  // Use a simple random id
  const newSessionId = Math.random().toString(36).slice(2, 10);
  const sess = createSession(bot, time_ms || 500);
  sessions.set(newSessionId, sess);

  // Send the move request
  return sendToSession(sess, black, white, time_ms, is_black_turn, (err, parsed) => {
    if (err) return res.status(500).json({error: 'bot error', details: err.toString()});
    // Return move plus session id so client can reuse
    const payload = Object.assign({}, parsed, {session: newSessionId});
    console.log('Responding with payload:', payload);
    res.json(payload);
  });
});

function sendToSession(sess, black, white, time_ms, is_black_turn, cb) {
  // push callback and write a single line command
  sess.pending.push(cb);
  sess.refreshTimeout();
  const line = `${String(black || 0)} ${String(white || 0)} ${String(time_ms || 500)} ${is_black_turn ? 1 : 0}` + '\n';
  try {
    sess.proc.stdin.write(line);
  } catch (e) {
    const cb2 = sess.pending.pop();
    if (cb2) cb2(e);
  }
}

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});

// Close a persistent bot session explicitly
app.post('/api/bot/close', (req, res) => {
  const {session} = req.body;
  if (!session) return res.status(400).json({error: 'no session'});
  console.log('Existing sessions:', Array.from(sessions.keys()));
  const sess = sessions.get(session);
  if (!sess) return res.status(404).json({error: 'unknown session'});
  try {
    sess.proc.kill();
    sessions.delete(session);
    return res.json({ok: true});
  } catch (e) {
    return res.status(500).json({error: e.toString()});
  }
});
