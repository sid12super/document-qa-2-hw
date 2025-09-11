import streamlit as st
import lab1  # Renamed from lab1
import lab2  # Renamed from lab2
import lab3  # Renamed from lab3

def main():
    st.set_page_config(page_title="HW Manager", page_icon="📚")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "HW1", "HW2"])

    # Render the selected page
    if page == "Home":
        # Landing page content
        st.title("Welcome to the HW Manager")
        st.write(
            "This multi-page app allows you to explore different homework assignments. "
            "Use the sidebar to navigate to 'HW1' or 'HW2'!"
        )
    elif page == "HW1":
        lab1.main()  # Call the `main` function from HW1.py
    elif page == "HW2":
        lab2.main()  # Call the `main` function from HW2.py
    elif page == "HW3":
        lab3.main()  # Call the `main` function from HW3.py

if __name__ == "__main__":
    main()



