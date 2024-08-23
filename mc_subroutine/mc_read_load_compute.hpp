//
// Created by polya on 8/23/24.
//

#ifndef MC_READ_LOAD_COMPUTE_HPP
#define MC_READ_LOAD_COMPUTE_HPP
#include "../potentialFunction/potentialFunctionPrototype.hpp"
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/object/pickle_support.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <cfenv> // for floating-point exceptions
#include <chrono>
#include <cstdlib>
#include <cxxabi.h>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>
namespace fs = boost::filesystem;
const auto PI=M_PI;

class mc_computation
{
public:
    mc_computation(const std::string &cppInParamsFileName): e2(std::random_device{}()),distUnif01(0.0, 1.0)
    {

        std::ifstream file(cppInParamsFileName);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;

        int paramCounter = 0;

        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty()) {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);

            //read T
            if (paramCounter == 0) {
                iss >> T;
                if (T <= 0) {
                    std::cerr << "T must be >0" << std::endl;
                    std::exit(1);
                }//end if
                std::cout << "T=" << T << std::endl;
                this->beta = 1 / T;

                paramCounter++;
                continue;
            }//end reading T

            //read unit cell number
            if (paramCounter==1) {

                iss >> N;

                xVecInit = std::shared_ptr<double[]>(new double[4], std::default_delete<double[]>());
                yVecInit = std::shared_ptr<double[]>(new double[4], std::default_delete<double[]>());

                mcNum_1sweep=8;

                paramCounter++;
                continue;

            }

            //read coefficients
            if(paramCounter==2){
                iss>>coefsToPotFunc;
                paramCounter++;
                continue;

            }// end reading coefficients

            //read potential function name
            if(paramCounter==3){
                iss>>potFuncName;
                paramCounter++;
                continue;
            }//end reading potential function name


            //read initial values
            if(paramCounter==4){
                std::vector<double> values;
                std::string valueStr;

                while (std::getline(iss, valueStr, ',')) {
                    values.push_back(std::stod(valueStr));
                }

                for(int i=0;i<4;i++){
                    xVecInit[i]=values[i];
                }
                for(int i=4;i<8;i++)
                {
                    yVecInit[i-4]=values[i];
                }





                paramCounter++;
                continue;




            }//end reading initial values

            //read sweepToWrite
            if(paramCounter==5){
                iss>>sweepToWrite;
                paramCounter++;
                continue;
            }//end reading sweepToWrite

            //read newFlushNum
            if(paramCounter==6){
                iss>>newFlushNum;
                paramCounter++;
                continue;
            }//end reading newFlushNum

            //read sweepLastFile
            if(paramCounter==7){
                //if sweepLastFileStr is "-1", sweepLastFile uses the overflowed value
                //and sweepLastFile+1 will be 0
                iss>>sweepLastFile;
                paramCounter++;
                continue;
            }//end reading sweepLastFile

            //read TDirRoot
            if (paramCounter==8){
                iss>>TDirRoot;
                paramCounter++;
                continue;
            }//end reading TDirRoot


            //read U_dist_dataDir
            if(paramCounter==9){
                iss>>U_dist_dataDir;
                paramCounter++;
                continue;
            }//end reading U_dist_dataDir

            //read h
            if(paramCounter==10){
                iss>>h;
                paramCounter++;
                std::cout << "h=" << h << std::endl;
                //                this->M = std::pow(2.0 * PI, 0.5) * h * 1.001;
                //                std::cout<<"M="<<M<<std::endl;
                continue;
            }



        }//end while
        this->potFuncPtr = createPotentialFunction(potFuncName, coefsToPotFunc);
        potFuncPtr->init();

        this->varNum=8+1;
        try {
            this->U_dist_ptr= std::shared_ptr<double[]>(new double[sweepToWrite * varNum],
                                                        std::default_delete<double[]>());
        }

        catch (const std::bad_alloc &e) {
            std::cerr << "Memory allocation error: " << e.what() << std::endl;
            std::exit(2);
        } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            std::exit(2);
        }

        std::cout<<"xVecInit: \n";
        print_shared_ptr(xVecInit,4);

        std::cout<<"yVecInit: \n";
        print_shared_ptr(yVecInit,4);

        std::cout<<"sweepToWrite="<<sweepToWrite<<std::endl;
        std::cout<<"mcNum_1sweep="<<mcNum_1sweep<<std::endl;
        std::cout<<"newFlushNum="<<newFlushNum<<std::endl;
        std::cout<<"sweepLastFile+1="<<sweepLastFile+1<<std::endl;
        std::cout<<"TDirRoot="<<TDirRoot<<std::endl;
        std::cout<<"U_dist_dataDir="<<U_dist_dataDir<<std::endl;

        dist0_3=std::uniform_int_distribution<int>(0,3);

        // int random_number=dist0_3(e2);
        // std::cout << "Random number: " << random_number << std::endl;


    }//end constructor

public:

    ///
    /// @param x
    /// @param leftEnd
    /// @param rightEnd
    /// @param eps
    /// @return return a value within distance eps from x, on the open interval (leftEnd, rightEnd)
    double generate_uni_open_interval(const double &x, const double &leftEnd, const double &rightEnd, const double &eps);

    ///
    /// @param x proposed value
    /// @param y current value
    /// @param a left end of interval
    /// @param b right end of interval
    /// @param epsilon half length
    /// @return proposal probability S(x|y)
    double S_uni(const double &x, const double &y,const double &a, const double &b, const double &epsilon);

    void proposal_uni(const std::shared_ptr<double[]> & xVecCurr,const std::shared_ptr<double[]> &yVecCurr,
        const int&pos,
        std::shared_ptr<double[]>&xVecNext, std::shared_ptr<double[]>&yVecNext);


    double acceptanceRatio_uni(const std::shared_ptr<double[]> & xVecCurr,const std::shared_ptr<double[]> & yVecCurr,
        const std::shared_ptr<double[]> & xVecNext, const std::shared_ptr<double[]> &yVecNext,
        const int &pos,const double &UCurr, double &UNext);

    void saveLastData2Csv(const std::shared_ptr<double[]>& array, const  int& arraySize, const std::string& filename, const int& numbersPerRow);

    void save_array_to_pickle_one_column(double *ptr, const int& startingInd, std::size_t size,const int & numbersPerRow, const std::string& filename);
    std::string generate_varName(const int &ind,const int &numbersPerRow);

    void execute_mc_one_sweep(std::shared_ptr<double[]>&xVecCurr,std::shared_ptr<double[]>& yVecCurr,
        std::shared_ptr<double[]>& xVecNext,std::shared_ptr<double[]>& yVecNext,
        const int &fls, const int& swp);

    void execute_mc(const std::shared_ptr<double[]> &xVec,const std::shared_ptr<double[]> &yVec,
        const int & sweepInit, const int & flushNum);

    void init_and_run();


    template<class T>
        void print_shared_ptr(const std::shared_ptr<T> &ptr,const int& size){
        if (!ptr) {
            std::cout << "Pointer is null." << std::endl;
            return;
        }

        for(int i=0;i<size;i++){
            if(i<size-1){
                std::cout<<ptr[i]<<",";
            }
            else{
                std::cout<<ptr[i]<<std::endl;
            }
        }

    }
public:
    double T;// temperature
    double beta;
    double h;// step size
    int sweepToWrite;
    int newFlushNum;
    int sweepLastFile;
    std::shared_ptr<potentialFunction> potFuncPtr;
    std::string TDirRoot;
    std::string U_dist_dataDir;
    std::shared_ptr<double[]> U_dist_ptr;
    int varNum;

    std::shared_ptr<double[]> xVecInit;
    std::shared_ptr<double[]> yVecInit;



    std::string coefsToPotFunc;
    std::string potFuncName;
    int N;//unit cell number
    int mcNum_1sweep;
    std::ranlux24_base e2;
    std::uniform_real_distribution<> distUnif01;
    std::uniform_int_distribution<int> dist0_3;
};



#endif //MC_READ_LOAD_COMPUTE_HPP
