// TILE COLOURS (USED)
FIELD_TILE_COLOUR = '#f9c549'; // 0
MOUNTAIN_TILE_COLOUR = '#534e5a'; // 1
HILLS_TILE_COLOUR = '#be753b'; // 2
PASTURES_TILE_COLOUR = '#84b54b'; // 3
FOREST_TILE_COLOUR = '#3d4e26'; // 4
DESERT_TILE_COLOUR = '#bf9b61'; // 5
WATER_TILE_COLOUR = '#039cdd'; // 6

// TILE COLOURS (UNSUSED)
CTHULHU_TILE_COLOUR = '#E85BEC'; // 7

// BOARD DIMENSIONS
BOARD_DIMS = [3, 4, 5, 4, 3];

// TYPE MAP
TYPE_MAP = [1, 3, 4, 0, 2, 3, 2, 0, 4, 5, 4, 1, 4, 1, 0, 3, 2, 0, 3];

// Setup function
function setup() { 
    createCanvas(720, 720);
    rectMode(CENTER);
} 
  
// Draw function (loops continiously during program execution)
function draw() { 

    background(WATER_TILE_COLOUR);

    stroke('#877578');
    strokeWeight(4);
    drawCatanGrid(200, 200, 60);
  
}

// Function to draw the Catan grid, using maths
function drawCatanGrid(centreXFirstPoint, centreYFirstPoint, radius) {

    let typeMapIndex = 0
    let x = centreXFirstPoint;
    let y = centreYFirstPoint;
    // Loops through each row of the board according to the BOARD_DIMS array
    for (let row = 0; row < BOARD_DIMS.length; row++) {
        // Recalculate x position based on board dimensions
        let hexesInRow = BOARD_DIMS[row];
        let e = hexesInRow - BOARD_DIMS[0];
        x = centreXFirstPoint - (e * (Math.sqrt(3) * radius / 2));
        // Draw each hexagon in the correct position
        for (let hex = 0; hex < hexesInRow; hex++) {
            let hexColour = returnColour(TYPE_MAP[typeMapIndex]);
            fill(hexColour);
            typeMapIndex = typeMapIndex + 1;
            drawHexagon(x, y, radius);
            x = x + (Math.sqrt(3) * radius);
        }
        // Recalculate y position for each row
        y = y + (3/4 * (radius * 2));
    }

}

// Function that draws a hexagon
// centreX and centreY determine where the hex is positioned
// The radius is squal to the circumradius of the hexagon
function drawHexagon(centreX, centreY, radius){

    // Call beginShape(), which instructs p5 to start drawing verticies of a more complex shape
    beginShape()

    // Make equiangular steps around the circle enclosing our hexagon
    for(let i = 1; i < 7; i += 1){

        // Calculate the radian angle using complex maths
        let a = ( PI / 180 ) * ( 60 * i - 30 )

        // Calculate the cartesian coordinates for a given angle and radius
        let x = centreX + radius * cos(a)
        let y = centreY + radius * sin(a)

        // Make the vertex using p5's vertex() method
        vertex(x, y) 

    }

    // Call endShape(CLOSE) to finish drawing the hexagon
    endShape(CLOSE)

}

// Return a colour hex value corresponding to the correct tile type
function returnColour(id) {

    switch (id) {
        case 0:
            return FIELD_TILE_COLOUR;
        case 1:
            return MOUNTAIN_TILE_COLOUR;
        case 2:
            return HILLS_TILE_COLOUR;
        case 3:
            return PASTURES_TILE_COLOUR;
        case 4:
            return FOREST_TILE_COLOUR;
        case 5:
            return DESERT_TILE_COLOUR;
        case 6:
            return WATER_TILE_COLOUR;
    }

}
