#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <stdio.h>

using std::cout;
using std::endl;

struct header_s
{
	std::string raw_file;		// raw data file name
	std::string source_name;	// source name
	int mach_id;			// machine id
	int tel_id;			// telescope id
	double ra;			// source right ascension
	double dec;			// source declinatio
	double az;			// azimuth angle in deg
	double zn;			// zenith angle in deg
	int data_type;			// data type ID
	double rdm;			// reference DM
	int nchans;			// number of channels
	double top_chn;			// frequency of the top channel MHz
	double band;			// channel bandwidth in MHz
	int nbeams;			// number of beams
	int ibeam;			// beam number
	int nbits;			// bits per sample
	double tstart;			// observation start time in MJD format
	double tsamp;			// sampling time in seconds
	int nifs;			// something
	size_t nsamps;			// number of time samples per channel
};

template<class DataType>
void SaveData(std::string outname, unsigned char *indata, unsigned char *outdata, int nchans, int tsamps, int nbits, header_s &head) {

	std::ofstream outfile(outname.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

	DataType* indatacast = reinterpret_cast<DataType*>(indata);

	for (int ii = 0; ii < nchans * tsamps; ii++) {
		outdata[ii] = static_cast<unsigned char>(indatacast[ii]);
	}

	cout << "Saving the header\n";

	int strlen;
	char field[60];

	strlen = 12;
	// header start - MUST be at the start!!
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "HEADER_START");
	outfile.write(field, strlen * sizeof(char));

	//telescope id
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "telescope_id");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.tel_id, sizeof(int));

	strlen = 11;
	// raw data file name
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "rawdatafile");
	outfile.write(field, strlen * sizeof(char));
	// need to restart after that
	strlen = head.raw_file.size();
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, head.raw_file.c_str());
	outfile.write(field, strlen * sizeof(char));

	strlen = 11;
	//source name
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "source_name");
	outfile.write(field, strlen * sizeof(char));
	// need to restart after that
	strlen = head.source_name.size();
	strcpy(field, head.source_name.c_str());
	outfile.write((char*)&strlen, sizeof(int));
	outfile.write(field, strlen * sizeof(char));

	strlen = 10;
	// machine id
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "machine_id");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.mach_id, sizeof(int));

	strlen = 9;
	//data type
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "data_type");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.data_type, sizeof(int));

	strlen = 8;
	// azimuth
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "az_start");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.az, sizeof(double));

	// zenith
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "za_start");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.zn, sizeof(double));

	strlen = 7;
	// source right ascension
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "src_raj");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.ra, sizeof(double));

	// source declination
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "src_dej");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.dec, sizeof(double));

	strlen = 6;
	// first sample time stamp
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "tstart");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.tstart, sizeof(double));

	// number of filterbank channels
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nchans");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.nchans, sizeof(int));

	// number of beams
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nbeams");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.nbeams, sizeof(int));

	strlen = 5;
	// sampling interval
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "tsamp");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.tsamp, sizeof(double));

	// bits per time sample
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nbits");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.nbits, sizeof(int));

	// reference dispersion measure
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "refdm");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.rdm, sizeof(double));

	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "ibeam");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.ibeam, sizeof(int));

	strlen = 4;
	// top channel frequency
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "fch1");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.top_chn, sizeof(double));

	// channel bandwidth
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "foff");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.band, sizeof(double));

	// number of if channels
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nifs");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&head.nifs, sizeof(int));

	strlen = 10;
	// header end - MUST be at the end!!
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "HEADER_END");
	outfile.write(field, strlen *sizeof(char));


	outfile.write(reinterpret_cast<char*>(outdata), nchans * tsamps);

	outfile.close();
}

int main(int argc, char *argv[])
{
	if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
		cout << "Usage: convert <input file> <output file> <outbits>" << endl << endl;
                exit(EXIT_SUCCESS);
	}
	std::stringstream oss;
	oss << atoi(argv[1]);
	std::string inname, outname;
	inname = std::string(argv[1]);
	outname = std::string(argv[2]);
	unsigned int outbits = atoi(argv[3]);
	cout << "Reading file: " << inname << endl;
	std::ifstream inputfile(inname.c_str(), std::ifstream::in | std::ifstream::binary);

    cout << "Read the file..." << endl;

    std::string read_param;
    char field[60];

    int strlen;
	int maxtsamp;

	header_s header;

	while(true)		// go 4eva
	{
		inputfile.read((char *)&strlen, sizeof(int));
		inputfile.read(field, strlen * sizeof(char));
		field[strlen] = '\0';
		read_param = field;

//		cout << "Read something";

		if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached
		else if (read_param == "rawdatafile")		// need to read some long filename
		{
			inputfile.read((char *)&strlen, sizeof(int));		// reads the length of the raw data file name
			inputfile.read(field, strlen * sizeof(char));
			field[strlen] = '\0';
			header.raw_file = field;
		}
		else if (read_param == "source_name")		// need to read source name
		{
			inputfile.read((char *)&strlen, sizeof(int));
			inputfile.read(field, strlen * sizeof(char));
			field[strlen] = '\0';
			header.source_name = field;
		}
		else if (read_param == "machine_id")	inputfile.read((char *)&header.mach_id, sizeof(int));
		else if (read_param == "telescope_id")	inputfile.read((char *)&header.tel_id, sizeof(int));
		else if (read_param == "src_raj")	inputfile.read((char *)&header.ra, sizeof(double));
		else if (read_param == "src_dej")	inputfile.read((char *)&header.dec, sizeof(double));
		else if (read_param == "az_start")	inputfile.read((char *)&header.az, sizeof(double));
		else if (read_param == "za_start")	inputfile.read((char *)&header.zn, sizeof(double));
		else if (read_param == "data_type")	inputfile.read((char *)&header.data_type, sizeof(int));
		else if (read_param == "refdm")		inputfile.read((char *)&header.rdm, sizeof(double));
		else if (read_param == "nchans")	inputfile.read((char *)&header.nchans, sizeof(int));
		else if (read_param == "fch1")		inputfile.read((char *)&header.top_chn, sizeof(double));
		else if (read_param == "foff")		inputfile.read((char *)&header.band, sizeof(double));
		else if (read_param == "nbeams")	inputfile.read((char *)&header.nbeams, sizeof(int));
		else if (read_param == "ibeam")		inputfile.read((char *)&header.ibeam, sizeof(int));
		else if (read_param == "nbits")		inputfile.read((char *)&header.nbits, sizeof(int));
		else if (read_param == "tstart")	inputfile.read((char *)&header.tstart, sizeof(double));
		else if (read_param == "tsamp")		inputfile.read((char *)&header.tsamp, sizeof(double));
		else if (read_param == "nifs")		inputfile.read((char *)&header.nifs, sizeof(int));
	}

    cout << "Read the header..." << endl;
	size_t headendpos = inputfile.tellg();
	inputfile.seekg(0, inputfile.end);
	size_t fileendpos = inputfile.tellg();
	inputfile.seekg(headendpos - 4, inputfile.beg);
	maxtsamp = (fileendpos - headendpos) / header.nchans / (header.nbits / 8);

    char *head = new char[4];
	inputfile.read(head, 4);
    for (int ii = 0; ii < 4; ii++)
		cout << head[ii];
    cout << endl << endl;
	cout.flush();

    unsigned int tsamp;
	tsamp = maxtsamp;
	cout << header.nchans << " " << tsamp << " " << header.nbits << endl;
	unsigned int to_read = header.nchans * tsamp * header.nbits / 8;
	unsigned int to_save = header.nchans * tsamp * outbits / 8;
	cout << to_read << " " << to_save << endl;
    unsigned char *indata = new unsigned char[to_read];
	unsigned char *outdata = new unsigned char[to_save];
    cout << "Reading some data now..." << endl;
	inputfile.read(reinterpret_cast<char*>(indata), to_read);

	int nbits = header.nbits;
	header.nbits = outbits;

	switch(nbits) {
		case 8:		SaveData<unsigned char>(outname, indata, outdata, header.nchans, tsamp, header.nbits, header);
					break;
		case 32: 	SaveData<float>(outname, indata, outdata, header.nchans, tsamp, header.nbits, header);
					break;
		default:	cout << "Value of bits currently not supported" << endl;
					break;
	}

    inputfile.close();

	delete [] indata;
	delete [] outdata;

	return 0;
}
