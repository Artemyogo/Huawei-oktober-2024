#include <bits/stdc++.h>
#include <unistd.h>

using namespace std;

int main(){
    system("cd ../../build && make all");
    for(int it = 0; it < 10000000000; it++){
        if(it % 100 == 0)
            cout << it << endl;
        string command = "cd ../../build && ./generator 100 50 100000 " + to_string(it) + " | ./main";
        system(command.c_str());
    }
}
