"use client";

// Note: This file seems to be the main page (e.g., src/app/page.tsx)
// It imports and uses the ProfileChat component.

import ProfileChat from '@/components/ProfileChat'; // Assuming ProfileChat is in components folder
import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';

// --- Amplify Configuration ---

// **FIXED**: Add runtime check for environment variables
const userPoolId = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID;
const userPoolClientId = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID;

// Check if the required environment variables are set
if (!userPoolId || !userPoolClientId) {
  // Throw an error during initialization if config is missing
  // This prevents the app from running with incomplete Amplify settings
  throw new Error("Required Cognito environment variables (NEXT_PUBLIC_COGNITO_USER_POOL_ID, NEXT_PUBLIC_COGNITO_CLIENT_ID) are not defined. Check your Amplify build environment variables.");
}

// Configure Amplify using the validated variables
Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: userPoolId, // Now guaranteed to be string if code proceeds
      userPoolClientId: userPoolClientId, // Now guaranteed to be string
      // region: process.env.NEXT_PUBLIC_AWS_REGION // Optional: Add region if needed
    }
  }
  // Add other Amplify categories here if needed (API, Storage, etc.)
});


// --- Authenticator Form Customization (Optional) ---
const formFields = {
  signUp: {
    email: {
      order: 1,
      isRequired: true,
      label: "Email Address",
      placeholder: "Enter your email"
    },
    // phone_number: { // Example: Uncomment if phone number is required in Cognito
    //   order: 2,
    //   isRequired: true,
    //   label: "Phone Number",
    //   placeholder: "Enter phone (e.g., +11234567890)"
    // },
    username: { // Keep if using username sign-in/sign-up
       order: 3,
       // isRequired: true // Usually true if username is primary identifier
    },
    password: {
      order: 4
    },
    confirm_password: {
      order: 5
    }
  },
  // Example: Customizing Sign In Header Text
  // signIn: {
  //   Header() {
  //     return <h2>Sign In to Profile Chat</h2>;
  //   },
  // }
};

// --- Main Page Component ---
export default function Home() {
  return (
    // Wrap the core application content with the Authenticator
    <Authenticator formFields={formFields} /* loginMechanisms={['email']} // Example: Specify login mechanism */ >
      {/* Render function provides signOut method and user object */}
      {({ signOut, user }) => (
        <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-8 lg:p-12 bg-gray-100 dark:bg-gray-900">
          <div className="w-full max-w-4xl flex flex-col items-end mb-4">
             {/* Display username and sign out button */}
             <div className="text-sm text-gray-600 dark:text-gray-300 mb-1">
               Logged in as: {user?.username || user?.signInDetails?.loginId || 'User'}
             </div>
             <button
                onClick={signOut}
                className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white text-xs rounded shadow transition duration-150"
             >
                Sign out
             </button>
          </div>

          {/* Render the main ProfileChat component */}
          {/* It should now be rendered only for authenticated users */}
          <div className="w-full">
             {/* Pass user info down if ProfileChat needs it */}
             <ProfileChat /* userId={user?.userId} username={user?.username} */ />
          </div>

        </main>
      )}
    </Authenticator>
  );
}
