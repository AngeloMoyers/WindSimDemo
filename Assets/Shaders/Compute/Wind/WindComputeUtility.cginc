#ifndef WIND_COMPUTE_LIB
#define WIND_COMPUTE_LIB

int CoordToIndex(int2 coord) {
    return clamp(coord.x, 0, uResolutionX - 1 ) + clamp(coord.y, 0, uResolutionY - 1 ) * uResolutionX;
}

int CoordToIndex(int3 coord) {
    return (coord.z * (uResolutionX) * (uResolutionY)) + (coord.y * (uResolutionX)) + coord.x;
}

#endif