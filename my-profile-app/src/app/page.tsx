// app/page.tsx
import ProfileChat from '@/components/ProfileChat'; // Adjust the import path if your components folder is elsewhere

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center">
      {/* You can add other elements to your page here */}
      <ProfileChat />
      {/* You can add other elements here too */}
    </main>
  );
}