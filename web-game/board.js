// TILE COLOURS (USED)
FIELD_TILE_COLOUR = '#ECDF5B';
MOUNTAIN_TILE_COLOUR = '#5B69EC';
HILLS_TILE_COLOUR = '#EC835B';
PASTURES_TILE_COLOUR = '#B3EC5B';
FOREST_TILE_COLOUR = '#68EC5B';
DESERT_TILE_COLOUR = '#ECC35B';
WATER_TILE_COLOUR = '#5BCFEC';

// TILE COLOURS (UNSUSED)
CTHULHU_TILE_COLOUR = '#E85BEC';

// BOARD DIMENSIONS
BOARD_DIMS = [3, 4, 5, 4, 3];

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
