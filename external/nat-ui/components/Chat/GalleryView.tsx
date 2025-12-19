import React from 'react';
import { Message } from '@/types/chat';
import { IconMap, IconGridDots } from '@tabler/icons-react';

interface GalleryViewProps {
  messages: Message[];
}

export const GalleryView: React.FC<GalleryViewProps> = ({ messages }) => {
  const imageMessages = messages.filter((message) => {
    return message.role === 'assistant' && (message.content.includes('![Annotated Image]') || message.content.includes('traffic_info'));
  });

  const extractData = (content: string) => {
    const imageMatch = content.match(/!\[Annotated Image\]\((.*?)\)/);
    // Fallback to finding any image if annotated one is missing, or use a placeholder
    const imageSrc = imageMatch ? imageMatch[1] : '';

    // More robust regex for congestion
    const congestionMatch = content.match(/\|\s*(?:🚦\s*)?拥堵等级\s*\|\s*(.*?)\s*\|/) || content.match(/congestion['"]?:\s*['"](.*?)['"]/);
    const congestion = congestionMatch ? congestionMatch[1].trim() : '未知';

    // More robust regex for vehicle count
    const vehicleCountMatch = content.match(/\|\s*(?:🚗\s*)?机动车数\s*\|\s*(\d+)\s*\|/) || content.match(/vehicle_count['"]?:\s*(\d+)/);
    const vehicleCount = vehicleCountMatch ? vehicleCountMatch[1] : '0';

    const timeMatch = content.match(/图片时间:\s*(.*?)\n/) || content.match(/timestamp['"]?:\s*['"](.*?)['"]/);
    const time = timeMatch ? timeMatch[1] : '';

    return { imageSrc, congestion, vehicleCount, time };
  };

  return (
    <div className="p-4 overflow-y-auto h-full">
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {imageMessages.map((message, index) => {
          const { imageSrc, congestion, vehicleCount, time } = extractData(message.content);
          
          // If no image source found, try to find the original input image from the user message before this one
          let displayImage = imageSrc;
          if (!displayImage) {
             // This is a simplification; ideally we'd look up the conversation history
             // For now, we'll show a placeholder if no image is found
          }

          if (!displayImage && !congestion) return null;

          return (
            <div key={index} className="border rounded-lg p-4 bg-white dark:bg-[#444654] shadow-sm hover:shadow-md transition-shadow">
              <div className="aspect-video relative mb-2 overflow-hidden rounded bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
                {displayImage ? (
                    <img src={displayImage} alt="Analysis Result" className="object-cover w-full h-full" 
                         onError={(e) => {
                             // Fallback for local paths that can't be loaded
                             (e.target as HTMLImageElement).style.display = 'none';
                             (e.target as HTMLImageElement).parentElement!.innerHTML = '<span class="text-xs text-gray-500">Image not accessible</span>';
                         }}
                    />
                ) : (
                    <span className="text-xs text-gray-500">No Image</span>
                )}
              </div>
              <div className="text-sm">
                <div className="font-bold mb-1 flex justify-between items-center">
                  <span>{time || '未知时间'}</span>
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    congestion.includes('畅通') ? 'bg-green-100 text-green-800' :
                    congestion.includes('拥堵') ? 'bg-red-100 text-red-800' :
                    'bg-yellow-100 text-yellow-800'
                  }`}>
                    {congestion}
                  </span>
                </div>
                <div className="text-gray-600 dark:text-gray-300">
                  <div>🚗 车辆: {vehicleCount}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {imageMessages.length === 0 && (
        <div className="text-center text-gray-500 mt-10">
          <IconGridDots size={48} className="mx-auto mb-2 opacity-50" />
          <p>暂无图像分析记录</p>
        </div>
      )}
    </div>
  );
};
