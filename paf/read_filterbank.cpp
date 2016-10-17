#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <stdio.h>

using std::cout;
using std::endl;

template<class DataType>
void SaveData(std::string outname, unsigned char *data, int nchans, int tsamps, int nbits) {

	std::ofstream outfile(outname.c_str(), std::ofstream::out | std::ofstream::trunc);

	DataType *outdata = reinterpret_cast<DataType*>(data);

	for (int ii = 0; ii < nchans * tsamps; ii++) {
		outfile << (unsigned int)data[ii] << endl;
	}
	outfile.close();
}

int main(int argc, char *argv[])
{
	if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
		cout << "Usage: read_filterbank <input file> <output file> <samples to read>" << endl << endl;
                exit(EXIT_SUCCESS);
	}
	std::stringstream oss;
	oss << atoi(argv[1]);
	std::string inname, outname;
	inname = std::string(argv[1]);
	outname = std::string(argv[2]);
	cout << "Reading file: " << inname << endl;
	std::ifstream inputfile(inname.c_str(), std::ifstream::in | std::ifstream::binary);

    cout << "Read the file..." << endl;

    std::string read_param;
    char field[60];

    int strlen;

	int inbits;
	int maxtsamp;
	int nchans;

	double tdouble;
	int tint;
	while(true)		// go 4eva
	{
		inputfile.read((char *)&strlen, sizeof(int));
		inputfile.read(field, strlen * sizeof(char));
		field[strlen] = '\0';
		read_param = field;

		if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached
		else if (read_param == "rawdatafile")		// need to read some long filename
		{
			inputfile.read((char *)&strlen, sizeof(int));		// reads the length of the raw data file name
			inputfile.read(field, strlen * sizeof(char));
			field[strlen] = '\0';
		}
		else if (read_param == "source_name")		// need to read source name
		{
			inputfile.read((char *)&strlen, sizeof(int));
			inputfile.read(field, strlen * sizeof(char));
			field[strlen] = '\0';
		}
		else if (read_param == "machine_id")	inputfile.read((char *)&tint, sizeof(int));
		else if (read_param == "telescope_id")	inputfile.read((char *)&tint, sizeof(int));
		else if (read_param == "src_raj")	inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "src_dej")	inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "az_start")	inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "za_start")	inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "data_type")	inputfile.read((char *)&tint, sizeof(int));
		else if (read_param == "refdm")		inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "fch1")		inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "foff")		inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "nbeams")	inputfile.read((char *)&tint, sizeof(int));
		else if (read_param == "ibeam")		inputfile.read((char *)&tint, sizeof(int));
		else if (read_param == "tstart")	inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "tsamp")		inputfile.read((char *)&tdouble, sizeof(double));
		else if (read_param == "nifs")		inputfile.read((char *)&tint, sizeof(int));

		if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached

		else if (read_param == "nchans")	inputfile.read((char *)&nchans, sizeof(int));
		else if (read_param == "nbits")		inputfile.read((char *)&inbits, sizeof(int));
	}

    cout << "Read the header..." << endl;

	size_t headendpos = inputfile.tellg();
	inputfile.seekg(0, inputfile.end);
	size_t fileendpos = inputfile.tellg();
	inputfile.seekg(headendpos - 4, inputfile.beg);
	maxtsamp = (fileendpos - headendpos) / nchans / (inbits / 8);

    char *head = new char[4];
	inputfile.read(head, 4);
    for (int ii = 0; ii < 4; ii++)
		cout << head[ii];
    cout << endl << endl;
	cout.flush();

    int tsamp;
	tsamp = std::min(atoi(argv[3]), maxtsamp);
	size_t to_read = nchans * tsamp * inbits / 8;
    unsigned char *data = new unsigned char[to_read];
    cout << "Reading some data now..." << endl;
	inputfile.read(reinterpret_cast<char*>(data), to_read);

	switch(inbits) {
		case 8:		SaveData<unsigned char>(outname, data, nchans, tsamp, inbits);
					break;
		case 32: 	SaveData<float>(outname, data, nchans, tsamp, inbits);
					break;
		default:	cout << "Value of bits currently not supported" << endl;
					break;
	}

    inputfile.close();

	return 0;
}
