#include<stdio.h>
#include <unistd.h>
#include <pthread.h>


int counter = 0;


void *increment(void *arg)
{
	int i;
	for(i=0;i<10000;i++)	
        pthread_mutex_lock(&lock);
		counter++;
        pthread_mutex_unlock(&lock);
	}
	return NULL;
}

int main()
{

	pthread_t thread1;
	pthread_t thread2;

	pthread_create(&thread1, NULL, increment, NULL);
	pthread_create(&thread2, NULL, increment, NULL);

	pthread_join( thread1, NULL);
	pthread_join( thread2, NULL);

	printf("Counter is %d\n",counter);

	return 0;
}