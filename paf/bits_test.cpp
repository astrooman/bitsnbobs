#include <bitset>
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {

    unsigned char header[8];
    bool flag;

    for (int ii = 0; ii < 8; ii++)
        header[ii] = ii - 4;

    for (int ii = 0; ii < 8; ii++)
        cout << (int)header[ii] << " -> " << std::bitset<8>(header[ii]) << endl;

    short meshort;
    float mefloat;

    meshort = static_cast<short>((header[1] << 8) | header[5]);
    mefloat = static_cast<float>(static_cast<short>(header[5] | (header[1] << 8)));

    cout << std::bitset<16>(((header[1] << 8) | header[5])) << endl;
    cout << meshort << " -> " << std::bitset<16>(meshort) << endl;
    cout << mefloat << endl;

    return 0;
}
