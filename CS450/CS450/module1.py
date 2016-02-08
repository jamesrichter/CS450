import io

class SearchStream:

    @staticmethod
    def find_lines(keyword, stream):
        
        return_stream = []
        for line in stream:
	        if keyword in line:
   		        return_stream.append(line)
                
        return return_stream


print SearchStream.find_lines('bob', "asdfasdfasdf\nbob\nbobobobo")