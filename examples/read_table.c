#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


struct undistortion_table {
    uint32_t height;
    uint32_t width;
    double *src_coords;
};


int main(int argc, char *argv[])
{
    FILE *fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        perror("fopen failed");
        exit(1);
    }

    struct undistortion_table table;
    int nelems = 0;

    nelems = fread(&table.height, sizeof(uint32_t), 1, fp);
    if (nelems != 1) {
        perror("fread failed");
        exit(1);
    }

    nelems = fread(&table.width, sizeof(uint32_t), 1, fp);
    if (nelems != 1) {
        perror("fread failed");
        exit(1);
    }

    printf("%d %d\n", table.height, table.width);

    size_t data_size = sizeof(double) * 2 * table.width * table.height;
    table.src_coords = malloc(data_size);
    nelems = fread(table.src_coords, sizeof(double), 2 * table.width * table.height, fp);

    double (*src_coords) [table.width][2];
    src_coords = (double (*) [table.width][2]) table.src_coords;

    printf("%f %f\n", src_coords[0][0][0], src_coords[0][0][1]);
    printf("%f %f\n", src_coords[50][50][0], src_coords[25][25][1]);
    printf("%f %f\n", src_coords[25][50][0], src_coords[25][50][1]);

    fclose(fp);
    return 0;
}