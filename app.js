const express = require('express');
const cors = require('cors');
const { Server } = require('socket.io');

const app = express();
const server = require('http').createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

app.use(cors());


io.on('connection', (socket) => {
  console.log('a user connected');

  socket.on('message', (message) => {
    console.log('received json:', message);
    io.send(message);
  });

  socket.on('send_message', (json_data) => {
    io.emit('data', { data: json_data }, { broadcast: true, include_self: false });
  });

  socket.on('disconnect', () => {
    console.log('user disconnected');
  });
});

const PORT = 10001;
server.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});