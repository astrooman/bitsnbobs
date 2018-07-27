#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <stdio.h>

using std::cout;
using std::endl;
using std::string;

struct FilHeader {
    std::string rawfile;
    std::string sourcename;

    double az;                      // azimuth angle in deg
    double dec;                     // source declination
    double fch1;                    // frequency of the top channel in MHz
    double foff;                    // channel bandwidth in MHz
    double ra;                      // source right ascension
    double rdm;                     // reference DM
    double tsamp;                   // sampling time in seconds
    double tstart;                  // observation start time in MJD format
    double za;                      // zenith angle in deg

    int datatype;                  // data type ID
    int ibeam;                      // beam number
    int machineid;
    int nbeams;
    int nbits;
    int nchans;
    int nifs;                       // something, something, something, DAAARK SIIIDEEEE
    int telescopeid;
};

template<class DataType>
void SaveData(std::string outname, unsigned char *data, int nchans, int tsamps, int nbits) {

	int saved = 0;

	std::ofstream outfile(outname.c_str(), std::ofstream::out | std::ofstream::trunc);

	DataType *outdata = reinterpret_cast<DataType*>(data);

	for (int ii = 0; ii < nchans * tsamps; ii++) {
		outfile << (float)outdata[ii] << endl;
	}
	outfile.close();
}

int main(int argc, char *argv[])
{

	if (argc < 2) {
		cout << "No command line options provided!" << endl;
		cout << "Use read_filterbank -h or --help" << endl;
		exit(EXIT_FAILURE);
	}

	if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
		cout << "Usage: read_filterbank <input file> <output file> <samples to read>" << endl << endl;
		cout << "Command line options: -i - print out the header information and quit" << endl;
	    exit(EXIT_SUCCESS);
	}

	FilHeader header;
	memset(&header, 0, sizeof(header));

	bool printheader = false;

	for (int iarg = 0; iarg < argc; ++iarg) {
		cout << argv[iarg] << endl;
		if (string(argv[iarg]) == "-i") {
			printheader = true;
			break;
		}
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

	// TODO: Add proper command line support

	while(true)
	{
		inputfile.read((char *)&strlen, sizeof(int));
		inputfile.read(field, strlen * sizeof(char));
		field[strlen] = '\0';
		read_param = field;

		if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached
		else if (read_param == "rawdatafile") {
			inputfile.read((char *)&strlen, sizeof(int));		// reads the length of the raw data file name
			inputfile.read(field, strlen * sizeof(char));
			field[strlen] = '\0';
			header.rawfile = field;
		} else if (read_param == "source_name") {
			inputfile.read((char *)&strlen, sizeof(int));
			inputfile.read(field, strlen * sizeof(char));
			field[strlen] = '\0';
			header.sourcename = field;
		} else if (read_param == "machine_id") {
			inputfile.read((char *)&header.machineid, sizeof(int));
		} else if (read_param == "telescope_id") {
			inputfile.read((char *)&header.machineid, sizeof(int));
		} else if (read_param == "src_raj") {
			inputfile.read((char *)&header.ra, sizeof(double));
		} else if (read_param == "src_dej")	{
			inputfile.read((char *)&header.dec, sizeof(double));
		} else if (read_param == "az_start") {
			inputfile.read((char *)&header.az, sizeof(double));
		} else if (read_param == "za_start") {
			inputfile.read((char *)&header.za, sizeof(double));
		} else if (read_param == "data_type") {
			inputfile.read((char *)&header.datatype, sizeof(int));
		} else if (read_param == "refdm") {
			inputfile.read((char *)&header.rdm, sizeof(double));
		} else if (read_param == "fch1") {
			inputfile.read((char *)&header.fch1, sizeof(double));
		} else if (read_param == "foff") {
			inputfile.read((char *)&header.foff, sizeof(double));
		} else if (read_param == "nbeams") {
			inputfile.read((char *)&header.nbeams, sizeof(int));
		} else if (read_param == "ibeam") {
			inputfile.read((char *)&header.ibeam, sizeof(int));
		} else if (read_param == "tstart") {
			inputfile.read((char *)&header.tstart, sizeof(double));
		} else if (read_param == "tsamp") {
			inputfile.read((char *)&header.tsamp, sizeof(double));
		} else if (read_param == "nifs") {
			inputfile.read((char *)&header.nifs, sizeof(int));
		} else if (read_param == "nchans") {
			inputfile.read((char *)&header.nchans, sizeof(int));
		} else if (read_param == "nbits") {
			inputfile.read((char *)&header.nbits, sizeof(int));
		}
		if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached
	}

	nchans = header.nchans;
	inbits = header.nbits;

    cout << "Read the header..." << endl;

	if (printheader) {
		cout << "Information stored in the header: " << endl;
		cout << "Source name: " << header.sourcename << endl;
		cout << "Azimuth / zenith angle: " << header.az << " / " << header.za << endl;
		cout << "RA / DEC: " << header.ra << " / " << header.dec << endl;
		cout << "Top channel frequency: " << header.fch1 << "MHz" << endl;
		cout << "Channel bandwidth: " << header.foff << "MHz" << endl;
		cout << "Number of channels: " << header.nchans << endl;
		cout << "Sampling time: " << header.tsamp * 1000.0f * 1000.0f << "us" << endl;
		cout << "Number of bits per sample: " << header.nbits << endl;

		inputfile.close();
		exit(EXIT_SUCCESS);
	}

    size_t headendpos = inputfile.tellg();
    inputfile.seekg(0, inputfile.end);
    size_t fileendpos = inputfile.tellg();
    inputfile.seekg(headendpos - 4, inputfile.beg);
    maxtsamp = (fileendpos - headendpos) / nchans / (inbits / 8);

    cout << "Number of channels: " << nchans << endl;
    cout << "Number of time samples: " << maxtsamp << endl;
    cout << "Bits per sample: " << inbits << endl;

    char *head = new char[4];
	inputfile.read(head, 4);
    for (int ii = 0; ii < 4; ii++)
		cout << head[ii];
    cout << endl << endl;
	cout.flush();

    unsigned int tsamp;
	unsigned int savetsamp = std::min(atoi(argv[3]), maxtsamp);
	tsamp = std::min(savetsamp, 1024U);
	unsigned int stride = nchans * (inbits / 8);
    unsigned int to_read = nchans * tsamp * (inbits / 8);
    unsigned char *data = new unsigned char[to_read];
    cout << "Reading some data now..." << endl;

	std::ofstream outfile(outname.c_str(), std::ofstream::out | std::ofstream::trunc);
	unsigned int saved = 0;

	cout << std::setprecision(2) << std::fixed;

	while (saved < savetsamp) {

		inputfile.read(reinterpret_cast<char*>(data), to_read);

		if (inbits == 8) {
			unsigned char *outdata = reinterpret_cast<unsigned char*>(data);
			for (int isamp = 0; isamp < nchans * tsamp; isamp++) {
				outfile << (float)outdata[isamp] << endl;
			}
		} else if (inbits == 32) {
			float *outdata = reinterpret_cast<float*>(data);
			for (int isamp = 0; isamp < nchans * tsamp; isamp++) {
				outfile << (float)outdata[isamp] << endl;
			}
		} else {
			cout << "The value of " << inbits << " input bits currently not supported" << endl;
			break;
		}

		saved += tsamp;
		cout << (float)saved / (float)savetsamp * 100.0f << "% done    \r";
		cout.flush();

	}

	cout << endl;

  	inputfile.close();

	return 0;
}
