#include <algorithm>
#include <chrono>
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{

    const int iter = 10000000;
    char *h_in = new char[128 * 7 * 8];
    for (int ii = 0; ii < 128 * 7 * 8; ii++)
        h_in[ii] = ii % 127;
   
    char *h_other = new char[128 * 7 * 8];

    std::chrono::time_point<std::chrono::high_resolution_clock> copy_start, copy_end;
    std::chrono::duration<double> copy_elapsed;

    copy_start = std::chrono::high_resolution_clock::now();
    for (int ii = 0; ii < iter; ii++)
        std::copy(h_in, h_in + 128 * 7 * 8, h_other);
    copy_end = std::chrono::high_resolution_clock::now();

    copy_elapsed = copy_end - copy_start;
    cout << std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count() / (double)iter << "us per std::copy" << endl;
    cout << copy_elapsed.count() << endl;
    cout << copy_elapsed.count() / (double)iter << endl;

    return 0;

}
