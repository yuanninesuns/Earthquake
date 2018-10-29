var app = require('express')();

var http = require('http').Server(app);

var io = require('socket.io')(http);

var rootDir ='/home/matthias/Desktop/Proj/test/'
app.get('/', function(req, res){
	res.sendFile(rootDir + 'index.html');
});

app.get('/2',function(req,res){
    res.sendFile(rootDir + '2.html');
});



var dl = require('/home/matthias/Desktop/Proj/test/lib/delivery.server'),
    fs  = require('fs');

function getPath (Path,Name) {
        return Path+Name;
    }


io.on('connection', function(socket){

var delivery = dl.listen(socket);
  delivery.on('receive.success',function(file){
    var params = file.params;
    fs.writeFile(file.name,file.buffer, function(err){
      if(err){
	
        console.log('File could not be saved.');
      }else{
	
        console.log('File saved.');
	Path=getPath('Download',file.name)
	io.emit('FilePath',Path);
      };
    });
  });

    console.log('Intel saiko!');
    var i = 0;

    socket.on('update',function(data){
        io.emit('update',data);

    })
    socket.on('total',function(data){
        io.emit('total',data);

    })
    socket.on('disconnect',function(){
        console.log('Intel No.1');
    });
    socket.on('detected',function(data){
        io.emit('detected',data)
    })

    socket.on('chat message', function(msg){
        console.log('message: ' + msg);
        io.emit('chat message',msg);
    });
});




function sleep(milliseconds){
    var start = new Date().getTime();
    for (var i = 0; i < 1e8; i++) {
        if ((new Date().getTime() - start) > milliseconds){
            break;
        }
    }
}

http.listen(3000, function(){
	  console.log('listening on *:3000');
});


