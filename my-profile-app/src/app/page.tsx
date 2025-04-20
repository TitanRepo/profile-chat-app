"use client";

import ProfileChat from '@/components/ProfileChat';
import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID,
      userPoolClientId: process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID,
    }
  }
});

const formFields = {
  signUp: {
    email: {
      order: 1, // Define order of fields
      isRequired: true, // Explicitly mark as required in UI
      label: "Email Address", // Custom label
      placeholder: "Enter your email"
    },
    // phone_number: {
    //   order: 2,
    //   isRequired: true, // Mark as required if needed
    //   label: "Phone Number",
    //   placeholder: "Enter your phone number (e.g., +11234567890)"
    //   // Note: You might need additional configuration for phone number format/dial code
    // },
    // Include other fields like password, confirm_password as needed
    // Their order will default if not specified, or you can set them.
    username: { // If using username sign-in
       order: 3
    },
    password: {
      order: 4
    },
    confirm_password: {
      order: 5
    }
  },
  // You can also customize fields for signIn, confirmSignUp, etc.
};

export default function Home() {
  return (
    <Authenticator formFields={formFields}>
      {({ signOut, user }) => (
        <main>
          <h1>Welcome {user?.username}</h1>
          {/* Pass user info or signOut function down if needed */}
          <ProfileChat />
          <button onClick={signOut}>Sign out</button>
        </main>
      )}
    </Authenticator>
  );
}