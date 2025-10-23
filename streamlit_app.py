import streamlit as st
import lab1  # Renamed from lab1
import lab2  # Renamed from lab2
import lab3
import lab4  # Renamed from lab3
import lab5  # Renamed from lab4
import lab6

def main():
    st.set_page_config(page_title="HW Manager", page_icon="📚")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "lab1", "lab2", "lab3", "lab4", "lab5", "lab6"])

    # Render the selected page
    if page == "Home":
        # Landing page content
        st.title("Welcome to the lab Manager")
        st.write(
            "This multi-page app allows you to explore different homework assignments. "
            "Use the sidebar to navigate to each lab."
        )
    elif page == "lab1":
        lab1.main()  # Call the `main` function from lab1.py
    elif page == "lab2":
        lab2.main()  # Call the `main` function from lab2.py
    elif page == "lab3":
        lab3.main()  # Call the `main` function from lab3.py
    elif page == "lab4":
        lab4.main()  # Call the `main` function from lab4.py
    elif page == "lab5":
        lab5.main()  # Call the `main` function from lab5.py
    elif page == "lab6":
        lab6.main()  # Call the `main` function from lab6.py
if __name__ == "__main__":
    main()



