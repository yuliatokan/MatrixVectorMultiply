#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <mpi.h>

int ProcNum = 0; 
int ProcRank = 0; 

void RandomInitialization(double* pMatrix, double* pVector, int Size) {
	int i, j;
	srand(unsigned(clock()));
	for (i = 0; i < Size; i++) {
		pVector[i] = rand() / double(1000);
		for (j = 0; j < Size; j++)
			pMatrix[i * Size + j] = rand() / double(1000);
	}
}

void InitializationData(double*& pMatrix, double*& pVector,
	double*& pResult, double*& pProcRows, double*& pProcResult,
	int& Size, int& RowNum) {
	int RestRows; // Number of rows, that haven’t been distributed yet
	int i; // Loop variable
	if (ProcRank == 0) {
		do {
			printf("\nEnter size of the initial objects: ");
			fflush(stdout);
			scanf("%d", &Size);
			if (Size < ProcNum) {
				printf("Size of the objects must be greater than number of processes!\n ");
			}
		} while (Size < ProcNum);
	}
	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	RestRows = Size;
	for (i = 0; i < ProcRank; i++)
		RestRows = RestRows - RestRows / (ProcNum - i);
	RowNum = RestRows / (ProcNum - ProcRank);
	pVector = new double[Size];
	pResult = new double[Size];
	pProcRows = new double[RowNum * Size];
	pProcResult = new double[RowNum];
	if (ProcRank == 0) {
		pMatrix = new double[Size * Size];
		RandomInitialization(pMatrix, pVector, Size);
	}
}

void DistributionData(double* pMatrix, double* pProcRows, double* pVector,
	int Size, int RowNum) {
	int* pSendNum; 
	int* pSendInd; 
	int RestRows = Size; 
	MPI_Bcast(pVector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];
	
	RowNum = (Size / ProcNum);
	pSendNum[0] = RowNum * Size;
	pSendInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		RestRows -= RowNum;
		RowNum = RestRows / (ProcNum - i);
		pSendNum[i] = RowNum * Size;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
	}
	
	MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
		pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	delete[] pSendNum;
	delete[] pSendInd;
}

void MakeResult(double* pProcResult, double* pResult, int Size,
	int RowNum) {
	int i; 
	int* pReceiveNum; 
	int* pReceiveInd; 
	int RestRows = Size; 

	pReceiveNum = new int[ProcNum];
	pReceiveInd = new int[ProcNum];
	
	pReceiveInd[0] = 0;
	pReceiveNum[0] = Size / ProcNum;
	for (i = 1; i < ProcNum; i++) {
		RestRows -= pReceiveNum[i - 1];
		pReceiveNum[i] = RestRows / (ProcNum - i);
		pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
	}

	
	MPI_Allgatherv(pProcResult, pReceiveNum[ProcRank], MPI_DOUBLE, pResult,
		pReceiveNum, pReceiveInd, MPI_DOUBLE, MPI_COMM_WORLD);
	
	delete[] pReceiveNum;
	delete[] pReceiveInd;
}

void ParallelMatrixVectorCalculation(double* pProcRows, double* pVector, double*
	pProcResult, int Size, int RowNum) {
	int i, j; 
	for (i = 0; i < RowNum; i++) {
		pProcResult[i] = 0;
		for (j = 0; j < Size; j++)
			pProcResult[i] += pProcRows[i * Size + j] * pVector[j];
	}
}

void TerminationData(double* pMatrix, double* pVector, double* pResult,
	double* pProcRows, double* pProcResult) {
	if (ProcRank == 0)
		delete[] pMatrix;
	delete[] pVector;
	delete[] pResult;
	delete[] pProcRows;
	delete[] pProcResult;
}

void main(int* argc, char** argv) {
	double* pMatrix; 
	double* pVector; 
	double* pResult; 
	int Size; 
	double* pProcRows;
	double* pProcResult;
	int RowNum;
	double Start, Finish, Duration;
	MPI_Init(argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	InitializationData(pMatrix, pVector, pResult, pProcRows, pProcResult,
		Size, RowNum);
	Start = MPI_Wtime();
	DistributionData(pMatrix, pProcRows, pVector, Size, RowNum);
	ParallelMatrixVectorCalculation(pProcRows, pVector, pProcResult, Size, RowNum);
	MakeResult(pProcResult, pResult, Size, RowNum);
	Finish = MPI_Wtime();
	Duration = Finish - Start;
	if (ProcRank == 0) {
		printf("Time of execution = %f\n", Duration);
	}
	TerminationData(pMatrix, pVector, pResult, pProcRows, pProcResult);
	MPI_Finalize();
}