
all: ext  

ext: 
	@echo
	@echo "Building extension"
	@echo "=================="
	@echo	
	
	rm -f -r __init__.py*
	python3 setup.py build_ext --inplace --force
	touch __init__.py
	
install: 
	@echo 
	@echo "Install extension"
	@echo "=================="
	@echo
	
	rm -f -r __init__.py*
	python3 setup.py install 
	touch __init__.py
		
uninstall: 	
	@echo
	@echo "Cleaning installed files"
	@echo "=================="
	@echo
	
	rm -f -r /usr/lib64/python3.5/site-packages/DataBoxEngine*  
	rm -f -r /usr/lib64/python2.7/site-packages/DataBoxEngine*  
	 
    
clean:
	@echo
	@echo "Cleaning generated files"
	@echo "=================="
	@echo
	
	rm -f -r build/
	rm -f -r dist/	
	find . -name '*.py[co]' -delete
	find . -name __pycache__ -delete
	
	rm -f -r __init__.py*	
	python3 setup.py clean
	touch __init__.py
	
	rm -f DataBoxEngine.cpp DataBoxEngine*.so  
	rm -f -r DataBoxEngine.egg-info/ 
	
	 
 
