#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <stdio.h>

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
	if (std::string(argv[1]) == "-h") {
		cout << "Usage: read_filterbank <input file> <output file> <samples to read>" << endl << endl;
	}
	std::stringstream oss;
	oss << atoi(argv[1]);
	std::string inname, outname;
	inname = std::string(argv[1]);
	outname = std::string(argv[2]);
	cout << "Reading file: " << inname << endl;
	std::ifstream input_file(inname.c_str(), std::ifstream::in | std::ifstream::binary);

    std::string read_param;
    char field[60];

    int strlen;

    input_file.seekg(369, input_file.beg);

    char *head = new char[4];

	input_file.read(head, 4);

    for (int ii = 0; ii < 4; ii++)
		cout << head[ii] << " ";

    cout << endl << endl;
	cout.flush();

    unsigned int tsamp;
	tsamp = atoi(argv[3]);
	size_t to_read = 567 * tsamp * 4;
    float *data = new float[567 * tsamp];
    cout << "Reading some data now..." << endl;
	input_file.read(reinterpret_cast<char*>(data), to_read);

    for (int ii = 0; ii < 21; ii++)
		cout << data[ii] << " ";
 	cout << endl << endl;

    std::ofstream out_file(outname.c_str(), std::ofstream::out | std::ofstream::trunc);

    for (int ii = 0; ii < 567 * tsamp; ii++) {
		out_file << data[ii] << endl;
	}

    input_file.close();
    out_file.close();

	return 0;
}
