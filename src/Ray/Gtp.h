#ifndef _GTP_H_
#define _GTP_H_

/////////////
//    //
/////////////

//  GTP
const int GTP_COMMAND_SIZE = 64;

//  GTP
const int BUF_SIZE = 256;

//  GTP
const int GTP_COMMANDS = 21;

//  ()
#define DELIM  " "

//  
#define PROGRAM_NAME  "Ray"

//  
#define PROGRAM_VERSION  "9.0.1"

//  GTP
#define PROTOCOL_VERSION  "2"


//////////////
//    //
//////////////

//  GTP
struct GTP_command_t {
  char command[GTP_COMMAND_SIZE];
  void (*function)();
};


////////////
//    //
////////////

#if defined (_WIN32)
#define STRCPY(dst, size, src) strcpy_s((dst), (size), (src))
#define STRTOK(src, token, next) strtok_s((src), (token), (next))
#else
#define STRCPY(dst, size, src) strcpy((dst), (src))
#define STRTOK(src, token, next) strtok((src), (token))
#endif

#define CHOMP(command) if(command[strlen(command)-1] == '\n') command[strlen(command)-1] = '\0'

// gtp
void GTP_main( void );

#endif
