#ifndef GILBERT
#define GILBERT

// adapted from gilbert2d.py, make it return a vector instead of an iterator. Original found at https://github.com/jakubcerveny/gilbert

#include <cstddef> //size_t
#include <vector>
#include <stdlib.h> // abs
#include <iostream>


int sgn(int x) {return x < 0 ? -1 : (x == 0 ? 0 : 1);} //map negative: -1, 0: 0, positives: 1


std::vector<std::vector<std::size_t>> generate2d(std::size_t x, std::size_t y, int ax, int ay, int bx, int by)
{
    int w = std::abs(ax + ay);
    int h = std::abs(bx + by);

    int dax = sgn(ax); 
    int day = sgn(ay);
    int dbx = sgn(bx);
    int dby = sgn(by);

    if (h == 1) {
        std::vector<std::vector<std::size_t>> gilbert(w, std::vector<std::size_t> (2, 0));
        for (std::size_t i = 0; i < w; i ++) {
            gilbert[i][0] = x + i * dax;
            gilbert[i][1] = y + i * day;
        }
        return gilbert;
    }
    
    if (w == 1) {
        std::vector<std::vector<std::size_t>> gilbert(h, std::vector<std::size_t> (2, 0));
        for (std::size_t i = 0; i < h; i ++) {
            gilbert[i][0] = x + i * dbx;
            gilbert[i][1] = y + i * dby;
        }
        return gilbert;
    }

    int ax2 = ax / 2;
    int ay2 = ay / 2;
    int bx2 = bx / 2;
    int by2 = by / 2;

    int w2 = std::abs(ax2 + ay2);
    int h2 = std::abs(bx2 + by2);

    std::vector<std::vector<std::size_t>> gilbert;
    std::vector<std::vector<std::size_t>> gilbert1;

    if (2 * w > 3 * h) {
        if ((w2 % 2 == 1) && (w > 2)) {
            ax2 = ax2 + dax;
            ay2 = ay2 + day;
        }
        gilbert = generate2d(x, y, ax2, ay2, bx, by);
        gilbert1 = generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by);
        gilbert.insert(gilbert.end(), gilbert1.begin(), gilbert1.end());
        return gilbert;
    }

    else {
        std::vector<std::vector<std::size_t>> gilbert2;
        if ((h2 % 2 == 1) && (h > 2)) {
            bx2 = bx2 + dbx;
            by2 = by2 + dby;
        }
        gilbert = generate2d(x, y, bx2, by2, ax2, ay2);
        gilbert1 = generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2);
        gilbert2 = generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby), -bx2, -by2, -(ax-ax2), -(ay-ay2));
        gilbert.insert(gilbert.end(), gilbert1.begin(), gilbert1.end());
        gilbert.insert(gilbert.end(), gilbert2.begin(), gilbert2.end());
        return gilbert;
    }
}

std::vector<std::vector<std::size_t>> gilbert2d(std::size_t nx, std::size_t ny)
{
    if (nx >= ny) {return generate2d(0, 0, nx, 0, 0, ny);}
    else {return generate2d(0, 0, 0, nx, ny, 0);}
} 


// int main()  // debuggind main
// {
//     std::size_t nx = 100;
//     std::size_t ny = 20;
//     std::vector<std::vector<std::size_t>> gilbert = gilbert2d(nx, ny);
//     for (size_t k = 0; k < nx * ny; k++)
//     {
//         std::cout << gilbert[k][0] << ", " << gilbert[k][1] << std::endl;
//     }
// }

#endif
