//
// Created by polya on 8/23/24.
//
#include "mc_read_load_compute.hpp"


///
/// @param x
/// @param leftEnd
/// @param rightEnd
/// @param eps
/// @return return a value within distance eps from x, on the open interval (leftEnd, rightEnd)
double mc_computation::generate_uni_open_interval(const double& x, const double& leftEnd, const double& rightEnd,
                                                  const double& eps)
{
    double xMinusEps = x - eps;
    double xPlusEps = x + eps;

    double unif_left_end = xMinusEps < leftEnd ? leftEnd : xMinusEps;
    double unif_right_end = xPlusEps > rightEnd ? rightEnd : xPlusEps;

    //    std::random_device rd;
    //    std::ranlux24_base e2(rd());

    double unif_left_end_double_on_the_right = std::nextafter(unif_left_end, std::numeric_limits<double>::infinity());


    std::uniform_real_distribution<> distUnif(unif_left_end_double_on_the_right, unif_right_end);
    //[unif_left_end_double_on_the_right, unif_right_end)

    double xNext = distUnif(e2);
    return xNext;
}


///
/// @param x proposed value
/// @param y current value
/// @param a left end of interval
/// @param b right end of interval
/// @param epsilon half length
/// @return proposal probability S(x|y)
double mc_computation::S_uni(const double& x, const double& y, const double& a, const double& b, const double& epsilon)
{
    if (a < y and y < a + epsilon)
    {
        return 1.0 / (y - a + epsilon);
    }
    else if (a + epsilon <= y and y < b + epsilon)
    {
        return 1.0 / (2.0 * epsilon);
    }
    else if (b - epsilon <= y and y < b)
    {
        return 1.0 / (b - y + epsilon);
    }
    else
    {
        std::cerr << "value out of range." << std::endl;
        std::exit(10);
    }
}


void mc_computation::proposal_uni(const std::shared_ptr<double[]>& xVecCurr, const std::shared_ptr<double[]>& yVecCurr,
                                  const int& pos,
                                  std::shared_ptr<double[]>& xVecNext, std::shared_ptr<double[]>& yVecNext)
{
    double lm = potFuncPtr->getLm();

    double xVec_pos_new = generate_uni_open_interval(xVecCurr[pos], 0, lm, h);


    double yVec_pos_new = generate_uni_open_interval(yVecCurr[pos], 0, lm, h);

    std::memcpy(xVecNext.get(), xVecCurr.get(), 4 * sizeof(double));

    xVecNext[pos] = xVec_pos_new;

    std::memcpy(yVecNext.get(), yVecCurr.get(), 4 * sizeof(double));

    yVecNext[pos] = yVec_pos_new;
}


double mc_computation::acceptanceRatio_uni(const std::shared_ptr<double[]>& xVecCurr,
                                           const std::shared_ptr<double[]>& yVecCurr,
                                           const std::shared_ptr<double[]>& xVecNext,
                                           const std::shared_ptr<double[]>& yVecNext,
                                           const int& pos, const double& UCurr, double& UNext)
{
    double lm = potFuncPtr->getLm();

    UNext = (*potFuncPtr)(xVecNext.get(), yVecNext.get(), pos);

    double numerator = -this->beta * UNext;
    double denominator = -this->beta * UCurr;
    double R = std::exp(numerator - denominator);

    // std::cout<<"computing S_x"<<std::endl;
    // std::cout<<"xVecCurr[pos]="<<xVecCurr[pos]<<std::endl;
    // std::cout<<"xVecNext[pos]="<<xVecNext[pos]<<std::endl;
    double S_xVecCurrNext = S_uni(xVecCurr[pos], xVecNext[pos], -lm, lm, h);
    double S_xVecNextCurr = S_uni(xVecNext[pos], xVecCurr[pos], -lm, lm, h);

    double ratio_x_pos = S_xVecCurrNext / S_xVecNextCurr;
    if (std::fetestexcept(FE_DIVBYZERO))
    {
        std::cout << "Division by zero exception caught." << std::endl;
        std::exit(15);
    }
    if (std::isnan(ratio_x_pos))
    {
        std::cout << "The result is NaN." << std::endl;
        std::exit(15);
    }

    R *= ratio_x_pos;

    double S_yVecCurrNext = S_uni(yVecCurr[pos], yVecNext[pos], -lm, lm, h);
    double S_yVecNextCurr = S_uni(yVecNext[pos], yVecCurr[pos], -lm, lm, h);

    double ratio_y_pos = S_yVecCurrNext / S_yVecNextCurr;

    if (std::fetestexcept(FE_DIVBYZERO))
    {
        std::cout << "Division by zero exception caught." << std::endl;
        std::exit(15);
    }
    if (std::isnan(ratio_y_pos))
    {
        std::cout << "The result is NaN." << std::endl;
        std::exit(15);
    }

    R *= ratio_y_pos;

    return std::min(1.0, R);
}


void mc_computation::saveLastData2Csv(const std::shared_ptr<double[]>& array, const int& arraySize,
                                      const std::string& filename, const int& numbersPerRow)
{
    //saves last row to csv
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    outFile << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::fixed;
    outFile << "U,x00,x01,x10,x11,y00,y01,y10,y11\n";
    for (int i = arraySize - numbersPerRow; i < arraySize; i++)
    {
        outFile << array[i];
        if ((i + 1) % numbersPerRow == 0)
        {
            outFile << '\n';
        }
        else
        {
            outFile << ',';
        }
    }
    outFile.close();
}


void mc_computation::save_array_to_pickle_one_column(double* ptr, const int& startingInd, std::size_t size,
                                                     const int& numbersPerRow, const std::string& filename)
{
    using namespace boost::python;
    try
    {
        Py_Initialize(); // Initialize the Python interpreter
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }

        // Debug output
        //        std::cout << "Python interpreter initialized successfully." << std::endl;

        // Import the pickle module
        object pickle = import("pickle");
        object pickle_dumps = pickle.attr("dumps");

        // Create a Python list from the C++ array
        list py_list;
        for (std::size_t i = startingInd; i < size; i += numbersPerRow)
        {
            py_list.append(ptr[i]);
        }

        // Serialize the list using pickle.dumps
        object serialized_array = pickle_dumps(py_list);

        // Extract the serialized data as a string
        std::string serialized_str = extract<std::string>(serialized_array);

        // Write the serialized data to a file
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing");
        }
        file.write(serialized_str.data(), serialized_str.size());
        file.close();

        // Debug output
        //        std::cout << "Array serialized and written to file successfully." << std::endl;
    }
    catch (const error_already_set&)
    {
        PyErr_Print();
        std::cerr << "Boost.Python error occurred." << std::endl;
    } catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    if (Py_IsInitialized())
    {
        Py_Finalize(); // Finalize the Python interpreter
    }
}


std::string mc_computation::generate_varName(const int& ind, const int& numbersPerRow)
{
    if (ind == 0)
    {
        return "U";
    } //end U

    if (ind == 1)
    {
        return "x00";
    } //end x00

    if (ind == 2)
    {
        return "x01";
    } //end x01

    if (ind == 3)
    {
        return "x10";
    } //end x10

    if (ind == 4)
    {
        return "x11";
    } //end x11

    if (ind == 5)
    {
        return "y00";
    } //end y00

    if (ind == 6)
    {
        return "y01";
    } //end y01

    if (ind == 7)
    {
        return "y10";
    } // end y10

    if (ind == 8)
    {
        return "y11";
    } //end y11

    std::exit(16);
}


void mc_computation::execute_mc_one_sweep(std::shared_ptr<double[]>& xVecCurr, std::shared_ptr<double[]>& yVecCurr,
                                          std::shared_ptr<double[]>& xVecNext, std::shared_ptr<double[]>& yVecNext,
                                          const int& fls, const int& swp)
{
    double UFull = potFuncPtr->potentialFull(xVecCurr.get(), yVecCurr.get());
    //next U
    double UNext;
    double UCurr;
    for (int j = 0; j < 4; j++)
    {
        //one mc in sweep
        int pos = dist0_3(e2); // position to change the value
        //propose next
        this->proposal_uni(xVecCurr, yVecCurr, pos, xVecNext, yVecNext);

        //next U

        UCurr = (*potFuncPtr)(xVecCurr.get(), yVecCurr.get(), pos);

        //accept-reject

        double r = this->acceptanceRatio_uni(xVecCurr, yVecCurr, xVecNext, yVecNext, pos, UCurr, UNext);
        double u = distUnif01(e2);
        double UCurrCpy = UCurr;

        if (u <= r)
        {
            std::memcpy(xVecCurr.get(), xVecNext.get(), 4 * sizeof(double));
            std::memcpy(yVecCurr.get(), yVecNext.get(), 4 * sizeof(double));
            UCurr = UNext;
        } //end of accept-reject

        UFull += UCurr - UCurrCpy;
    } //end sweep for
    U_dist_ptr[swp * varNum + 0] = UFull;
    std::memcpy(U_dist_ptr.get() + swp * varNum + 1, xVecCurr.get(), 4 * sizeof(double));
    std::memcpy(U_dist_ptr.get() + swp * varNum + 5, yVecCurr.get(), 4 * sizeof(double));
}



void mc_computation::execute_mc(const std::shared_ptr<double[]> &xVec,const std::shared_ptr<double[]> &yVec,
        const int & sweepInit, const int & flushNum)
{

    std::shared_ptr<double[]> xVecCurr = std::shared_ptr<double[]>(new double[4], std::default_delete<double[]>());
    std::shared_ptr<double[]> xVecNext = std::shared_ptr<double[]>(new double[4], std::default_delete<double[]>());

    std::memcpy(xVecCurr.get(), xVec.get(), 4* sizeof(double));

    std::shared_ptr<double[]> yVecCurr = std::shared_ptr<double[]>(new double[4], std::default_delete<double[]>());
    std::shared_ptr<double[]> yVecNext = std::shared_ptr<double[]>(new double[4], std::default_delete<double[]>());

    std::memcpy(yVecCurr.get(),yVec.get(),4*sizeof(double));

    int sweepStart = sweepInit;

    for (int fls = 0; fls < flushNum; fls++)
    { const auto tMCStart{std::chrono::steady_clock::now()};
        for (int swp = 0; swp < sweepToWrite; swp++)
        {
            execute_mc_one_sweep(xVecCurr,yVecCurr,xVecNext,yVecNext, fls, swp);
        }//end sweep for

        int sweepEnd = sweepStart + sweepToWrite - 1;
        std::string fileNameMiddle = "sweepStart" + std::to_string(sweepStart) + "sweepEnd" + std::to_string(sweepEnd);

        std::string out_U_distPickleFileName_pkl = this->U_dist_dataDir + "/" + fileNameMiddle + ".U_dist.pkl";

        std::string out_U_distPickleFileName_csv = this->U_dist_dataDir + "/" + fileNameMiddle + ".U_dist.csv";

        saveLastData2Csv(U_dist_ptr, sweepToWrite  * varNum, out_U_distPickleFileName_csv, varNum);
        for (int startingInd = 0; startingInd < varNum; startingInd++)
        {
            std::string varName = generate_varName(startingInd, varNum);

            std::string outVarPath = this->U_dist_dataDir + "/" + varName + "/";

            if (!fs::is_directory(outVarPath) || !fs::exists(outVarPath)) {
                fs::create_directories(outVarPath);
            }

            std::string outVarFile = outVarPath + "/" + fileNameMiddle + "." + varName + ".pkl";

            save_array_to_pickle_one_column(U_dist_ptr.get(), startingInd, sweepToWrite  * varNum, varNum,
                                                        outVarFile);

        }

        const auto tMCEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
        std::cout << "sweep " + std::to_string(sweepStart) + " to sweep " + std::to_string(sweepEnd) + ": "
                  << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
        sweepStart = sweepEnd + 1;
    }//end flush for loop

    std::cout << "mc executed for " << flushNum << " flushes." << std::endl;


}


void mc_computation::init_and_run()
{
this->execute_mc(xVecInit,yVecInit,sweepLastFile+1,newFlushNum);


}