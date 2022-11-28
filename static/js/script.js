const canvas = document.getElementById("canvas");
const clearEl = document.getElementById("clear");
const recogniseEl = document.getElementById("recognise");
const ctx = canvas.getContext("2d");

let size = 10;
let isPressed = false;
let color = "black";
let x = undefined;
let y = undefined;
let isEraser = false;

canvas.addEventListener("mousedown", (e) => {
    isPressed = true;

    x = e.offsetX;
    y = e.offsetY;
});

canvas.addEventListener("mouseup", (e) => {
    isPressed = false;

    x = undefined;
    y = undefined;
});

canvas.addEventListener("mousemove", (e) => {
    if (isPressed) {
        if (!isEraser) {
            const x2 = e.offsetX;
            const y2 = e.offsetY;
            drawCircle(x2, y2);
            drawLine(x, y, x2, y2);
            x = x2;
            y = y2;
        }
        else{
            mouseX = e.pageX - canvas.offsetLeft;
            mouseY = e.pageY - canvas.offsetTop;
            ctx.clearRect(mouseX, mouseY, size, size);
        }
    }
});

function drawCircle(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawLine(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = size * 2;
    ctx.stroke();
}

clearEl.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

recogniseEl.addEventListener("click",()=>{
    const base64CanvasD1 = canvas.toDataURL('image/png').replace('data:image/png;base64,', '');

    const data = {
      operation: base64CanvasD1,
    }

    $.ajax({
      url: '/predict',
      type:'POST',
      data,
    }).done(function(data) {
        let result = JSON.parse(data);
        console.log(result)
        //$('#operation-container').html(result.operation);
        $('#solution-container').html(result.solution);
    }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
        console.log(XMLHttpRequest);
        alert("error");
    })
})