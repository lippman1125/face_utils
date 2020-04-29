#include <fstream>
#include <iostream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cmath>
#include <dirent.h>
#include <string.h>
#include <strings.h>
#include <iomanip>
#include <sys/time.h>
#include "cluster.hpp"

using namespace std;
void split(std::string& s, std::string& delim,std::vector< std::string >& ret) {
    size_t last = 0;
    size_t index=s.find_first_of(delim,last);
    while (index!=std::string::npos) {
        //skip continous space
        if (index > last) {
            ret.push_back(s.substr(last,index-last));
        }
        last=index+1;
        index=s.find_first_of(delim,last);
    }

    if (index-last>0) {
        ret.push_back(s.substr(last,index-last));
    }
}

void search_files(string root, vector<string> &files){
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (root.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
                continue;
            files.push_back(root + "/" + string(ent->d_name));

        }
        closedir (dir);
    }
}

int get_descs(string &root, std::ifstream &facedir_file, vector<vector<float>>& face_descriptors) {
    string line;
    string line_desc;
    string delim=" ";
    vector<string> dirs;
    size_t size;
    vector<string> imgs_path;
    vector<float> face_descriptor;
    std::ifstream desc_file;

    face_descriptors.clear();
    if (std::getline(facedir_file, line)) {
        split(line, delim, dirs);
        cout <<"dirs size: "<<dirs.size()<<endl;
        for (vector<string>::iterator iter = dirs.begin(); iter != dirs.end(); iter++) {
            imgs_path.clear();
            cout<<"sub dir path: " << root + "/" + *iter << endl;
            search_files(root + "/" + *iter, imgs_path);
            for (size_t i = 0; i < imgs_path.size(); i++) {
                face_descriptor.clear();
                desc_file.open(imgs_path[i], ios::in);
                while(std::getline(desc_file, line_desc)) {
                    face_descriptor.push_back(std::stof(line_desc, &size));
                }
                face_descriptors.push_back(face_descriptor);
                desc_file.close();
            }
        }
        return true;
    } else {
        return false;
    }
}


int main(int argc, char * * argv) {
    if (argc != 3) {
        cout <<"face_cluster facedir_list.txt root_dir" << endl;
        return 0;
    }
    std::srand (unsigned(std::time(0)));
    ifstream facedir_list(argv[1]);
    if (!facedir_list) {
       cout<<"file :"<<argv[1] << " doesn't exit"<<endl;
       return 0;
    }
    class FaceCluster face_cluster(0.35);

    vector<vector<float>> face_descriptors;
    pair<unsigned long, vector<unsigned long>> labels;
    string root(argv[2]);

    while(get_descs(root, facedir_list, face_descriptors)) {
        for (size_t i = 0; i < face_descriptors.size(); i++) {
            cout << "___desc no."<< i << "___" << endl;
            for (size_t j = 0; j < face_descriptors[i].size(); j++) {
                cout<< face_descriptors[i][j] << endl;
            }
        }
        labels = face_cluster.Cluster(face_descriptors);
        cout<<"cluster num: "<<labels.first<<endl;
        for (size_t i = 0; i < labels.second.size(); i++) {
            cout<<labels.second[i]<<" ";
            if ((i+1)%8==0) {
                cout<<endl;
            }
        }
        break;
    }
    return 0;
}
